# make_dataset_PROBAR.py (CLI)
from __future__ import annotations
from pathlib import Path
import argparse
import random, math
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import fftconvolve

SR = 16000
CLASSES = ["CONTINUAR","REPETIR","SALIR","PRINCIPAL","ACOPLAR","CANCELAR","LEVANTADO","FONDO"]

# --------------------- utilidades WAV ---------------------
def load_wav_mono(path: str, sr: int = SR) -> np.ndarray:
    x, _ = librosa.load(path, sr=sr, mono=True)
    x = x - np.mean(x)
    return x.astype(np.float32)

def save_wav(path: str, x: np.ndarray, sr: int = SR):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, sr, subtype="PCM_16")

# --------------------- aumentaciones ----------------------
# Estiramiento del tiempo random:
def rand_time_shift(x: np.ndarray, max_ms=100) -> np.ndarray:
    max_samp = int(SR * max_ms / 1000)
    shift = random.randint(-max_samp, max_samp)
    if shift > 0:
        return np.r_[np.zeros(shift, dtype=x.dtype), x[:-shift]]
    elif shift < 0:
        return np.r_[x[-shift:], np.zeros(-shift, dtype=x.dtype)]
    return x

# Cambiar velocidad (y pitch) con resample:
def rand_time_stretch(x: np.ndarray, low=0.9, high=1.1) -> np.ndarray:
    """
    Stretch sobre waveform usando resample:
    - Elegimos un rate en [low, high].
    - Re-sampleamos a SR*rate para cambiar duración.
    - Volvemos a SR para mantener consistencia global.
    """
    rate = random.uniform(low, high)
    y = librosa.resample(x, orig_sr=SR, target_sr=int(SR * rate))
    y = librosa.resample(y, orig_sr=int(SR * rate), target_sr=SR)
    return y.astype(np.float32)

# Cambiar pitch sin afectar duración:
def rand_pitch_shift(x: np.ndarray, semitones=1) -> np.ndarray:
    n_steps = random.uniform(-semitones, semitones)
    # librosa >=0.10 requiere argumentos con nombre
    return librosa.effects.pitch_shift(y=x, sr=SR, n_steps=n_steps).astype(np.float32)

# Recortar o rellenar con ceros a 1 s:
def pad_or_trim_1s(x: np.ndarray, dur_s=1.0) -> np.ndarray:
    L = int(SR * dur_s)
    if len(x) >= L:
        start = random.randint(0, len(x) - L)
        return x[start:start+L]
    pad = L - len(x)
    left = pad // 2
    right = pad - left
    return np.pad(x, (left, right), mode="constant")

# Mezclar con ruido a SNR dado:
def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(speech):
        reps = math.ceil(len(speech) / len(noise))
        noise = np.tile(noise, reps)
    noise = noise[:len(speech)]
    Ps = np.mean(speech**2) + 1e-12
    Pn = np.mean(noise**2) + 1e-12
    alpha = math.sqrt(Ps / (Pn * (10**(snr_db/10))))
    y = speech + alpha * noise
    mx = max(1e-8, np.max(np.abs(y)))
    if mx > 0.99:
        y = y / mx * 0.99
    return y.astype(np.float32)

# Aplicar reverberación con RIR:
def apply_reverb(x: np.ndarray, rir: np.ndarray, wet=0.3) -> np.ndarray:
    dry = x
    wet_sig = fftconvolve(x, rir, mode="full")[:len(x)]
    y = (1-wet)*dry + wet*wet_sig
    return y.astype(np.float32)

# Combinación de aumentaciones:
def random_augment(
    x: np.ndarray,
    p_shift=0.8,
    p_stretch=0.5,
    p_pitch=0.5,
    max_shift_ms=100,
    stretch_low=0.9,
    stretch_high=1.1,
    pitch_semitones=1.0,
) -> np.ndarray:
    if p_shift > 0 and random.random() < p_shift:
        x = rand_time_shift(x, max_ms=max_shift_ms)
    if p_stretch > 0 and random.random() < p_stretch:
        x = rand_time_stretch(x, low=stretch_low, high=stretch_high)
    if p_pitch > 0 and random.random() < p_pitch:
        x = rand_pitch_shift(x, semitones=pitch_semitones)
    return x

# ------------------- dataset helpers ----------------------
# Extraer segmento aleatorio de ruido (con repetición si es necesario):
def random_noise_segment(noises: list[np.ndarray], L: int) -> np.ndarray:
    n = random.choice(noises)
    if len(n) < L:
        reps = math.ceil(L/len(n))
        n = np.tile(n, reps)
    start = random.randint(0, len(n)-L)
    seg = n[start:start+L]
    return seg.copy()

# Cargar corpus de semillas (seeds) desde disco:
def load_corpus(root: str) -> dict[str, list[str]]:
    rootp = Path(root)
    items = {}
    for c in CLASSES:
        p = rootp / c
        items[c] = [str(x) for x in p.rglob("*.wav")]
    return items

# ------------------- generador principal -------------------
# Generar dataset con aumentaciones:
def make_dataset(
    seeds_root="seeds", # carpeta con subcarpetas por clase
    noises_root="noises", # carpeta con ruidos/charlas
    rirs_root="rirs", # carpeta con RIRs
    out_root="data", # carpeta de salida
    per_class=3000, # cantidad de ejemplos por clase (excluyendo FONDO)
    fondo_minutes=60, # minutos de FONDO (ruido/charlas)
    # aug params: 
    p_shift=0.8, # probabilidad de shift temporal
    p_stretch=0.5, # probabilidad de estiramiento temporal
    p_pitch=0.5, # probabilidad de cambio de pitch
    max_shift_ms=100, # shift máximo en ms
    stretch_low=0.9, # factor mínimo de estiramiento
    stretch_high=1.1, # factor máximo de estiramiento
    pitch_semitones=1.0, # semitonos máximos para pitch
    p_noise=0.9, # probabilidad de agregar ruido
    p_reverb=0.3, # probabilidad de agregar reverb
    snr_list=(20, 10, 5, 0), # lista de SNRs en dB
    use_fondo_as_noise=True, # usar seeds/FONDO como ruidos adicionales
):
    # cargar listas de archivos
    seeds = load_corpus(seeds_root)
    noise_files = list(Path(noises_root).rglob("*.wav"))
    rir_files = list(Path(rirs_root).rglob("*.wav"))

    # precargar ruidos/RIRs
    noises = [load_wav_mono(str(f)) for f in noise_files] if noise_files else []
    rirs   = [load_wav_mono(str(f)) for f in rir_files] if rir_files else []

    # usar seeds/FONDO como ruidos adicionales (opcional)
    if use_fondo_as_noise and "FONDO" in seeds and seeds["FONDO"]:
        for f in seeds["FONDO"]:
            noises.append(load_wav_mono(f))
        print(f"Usando {len(seeds['FONDO'])} archivos de seeds/FONDO como ruido adicional.")

    if not noises:
        # respaldo: ruido gaussiano de 30 s
        noises = [np.random.randn(SR * 30).astype(np.float32)]
        print("[WARN] No se encontraron ruidos externos; usando ruido gaussiano sintético.")

    # generar palabras clave
    for c in CLASSES:
        if c == "FONDO":
            continue
        out_dir = Path(out_root) / c
        out_dir.mkdir(parents=True, exist_ok=True)
        pool = seeds.get(c, [])
        if not pool:
            print(f"[WARN] Sin semillas para {c} en {seeds_root}/{c}")
            continue

        for i in range(per_class):
            wav_path = random.choice(pool)
            x = load_wav_mono(wav_path)  # semilla

            # aumentaciones temporales/pitch
            x = random_augment(
                x,
                p_shift=p_shift,
                p_stretch=p_stretch,
                p_pitch=p_pitch,
                max_shift_ms=max_shift_ms,
                stretch_low=stretch_low,
                stretch_high=stretch_high,
                pitch_semitones=pitch_semitones,
            )

            x = pad_or_trim_1s(x, 1.0)

            # ruido y/o reverb aleatorios
            if random.random() < p_noise and noises:
                snr = random.choice(list(snr_list))
                nseg = random_noise_segment(noises, len(x))
                x = mix_at_snr(x, nseg, snr)
            if random.random() < p_reverb and rirs:
                rir = random.choice(rirs)
                x = apply_reverb(x, rir, wet=0.3)

            save_wav(out_dir / f"{c.lower()}_{i:05d}.wav", x)

    # generar FONDO (trozos de ruido/charlas)
    out_fondo = Path(out_root) / "FONDO"
    out_fondo.mkdir(parents=True, exist_ok=True)
    total_secs = int(fondo_minutes * 60)
    num_clips = total_secs  # 1 s c/u
    for i in range(num_clips):
        seg = random_noise_segment(noises, SR)  # 1 s
        if random.random() < 0.2 and rirs:
            seg = apply_reverb(seg, random.choice(rirs), wet=0.3)
        save_wav(out_fondo / f"fondo_{i:05d}.wav", seg)

    print("✔ Dataset generado en", Path(out_root).resolve())

# --------------------------- CLI ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Generador de dataset KWS (ES) con aumentaciones")
    ap.add_argument("--seeds-root", type=str, default="seeds") # carpeta con subcarpetas por clase
    ap.add_argument("--noises-root", type=str, default="noises") # carpeta con ruidos/charlas
    ap.add_argument("--rirs-root", type=str, default="rirs") # carpeta con RIR (Room Impulse Response/Respuesta al Impulso de la Habitación)
    ap.add_argument("--out-root", type=str, default="data") # carpeta de salida
    ap.add_argument("--per-class", type=int, default=3000) # cantidad de ejemplos por clase (excluyendo FONDO)
    ap.add_argument("--fondo-minutes", type=float, default=60) # minutos de FONDO (ruido/charlas)

    ap.add_argument("--p-shift", type=float, default=0.8) # probabilidad de shift temporal
    ap.add_argument("--p-stretch", type=float, default=0.5) # probabilidad de estiramiento temporal
    ap.add_argument("--p-pitch", type=float, default=0.5) # probabilidad de cambio de pitch
    ap.add_argument("--max-shift-ms", type=int, default=100) # shift máximo en ms
    ap.add_argument("--stretch-low", type=float, default=0.9) # factor mínimo de estiramiento
    ap.add_argument("--stretch-high", type=float, default=1.1)# factor máximo de estiramiento
    ap.add_argument("--pitch-semitones", type=float, default=1.0) # semitonos máximos para pitch

    ap.add_argument("--p-noise", type=float, default=0.9) # probabilidad de agregar ruido
    ap.add_argument("--p-reverb", type=float, default=0.3) # probabilidad de agregar reverb
    ap.add_argument("--snr-list", type=str, default="20,10,5,0",
                    help="Lista de SNRs en dB separadas por coma (ej: '20,10,5,0')") # lista de SNRs en dB
    ap.add_argument("--use-fondo-as-noise", action="store_true",
                    help="Usar seeds/FONDO como ruidos adicionales") # usar seeds/FONDO como ruidos adicionales
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    snrs = tuple(float(x) for x in args.snr_list.split(",") if x.strip() != "")

    make_dataset(
        seeds_root=args.seeds_root,
        noises_root=args.noises_root,
        rirs_root=args.rirs_root,
        out_root=args.out_root,
        per_class=args.per_class,
        fondo_minutes=args.fondo_minutes,
        p_shift=args.p_shift,
        p_stretch=args.p_stretch,
        p_pitch=args.p_pitch,
        max_shift_ms=args.max_shift_ms,
        stretch_low=args.stretch_low,
        stretch_high=args.stretch_high,
        pitch_semitones=args.pitch_semitones,
        p_noise=args.p_noise,
        p_reverb=args.p_reverb,
        snr_list=snrs,
        use_fondo_as_noise=args.use_fondo_as_noise,
    )
