# features.py
from __future__ import annotations
from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt

# Clases del proyecto (ajustá el orden si querés otro mapeo)
CLASSES = [
    "ACOPLAR", "CANCELAR", "CONTINUAR", "FONDO",
    "LEVANTADO", "PRINCIPAL", "REPETIR", "SALIR",
]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------------
# Loader robusto de audio
# ---------------------------------------------------------------------
def _safe_load_audio(wav_path: str, sr_target: int) -> Tuple[torch.Tensor, int]:
    """
    Carga robusta de audio.
    @param wav_path: ruta al archivo WAV
    @param sr_target: frecuencia de muestreo destino
    @return: (wav[1, N] float32, sr)
    Orden de intento: TorchCodec -> soundfile -> librosa -> torchaudio.
    """
    # 1) TorchCodec
    try:
        from torchcodec.decoders import AudioDecoder
        dec = AudioDecoder(wav_path)
        audio, sr = dec.read_all()                  # np.ndarray [N] o [N, C]
        if audio.ndim == 1:
            audio = audio[:, None]                  # -> [N, 1]
        wav = torch.from_numpy(audio.T.astype("float32"))  # [C, N]
    except Exception:
        # 2) soundfile
        try:
            import soundfile as sf
            audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)  # [N, C]
            wav = torch.from_numpy(audio.T)                                  # [C, N]
        except Exception:
            # 3) librosa
            try:
                import librosa
                x, sr = librosa.load(wav_path, sr=None, mono=False)          # [N] o [C,N]
                if isinstance(x, list):
                    x = np.array(x)
                if x.ndim == 1:
                    x = x[None, :]                                           # [1, N]
                wav = torch.from_numpy(x.astype("float32"))                  # [C, N]
            except Exception:
                # 4) torchaudio (último recurso)
                wav, sr = torchaudio.load(wav_path)                          # [C, N]
                wav = wav.to(torch.float32)

    # A mono si es estéreo
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # [1, N]

    # Resample si hace falta
    if sr != sr_target:
        wav = torchaudio.functional.resample(wav, sr, sr_target)
        sr = sr_target

    # Quitar DC
    wav = wav - wav.mean()

    return wav, sr


# ---------------------------------------------------------------------
# WAV -> log-Mel (+ guardado automático en tensores/CLASE/*.pt)
# ---------------------------------------------------------------------
def wav_to_logmel_torch(
    wav_path: str,
    sr_target: int = 16000,
    n_mels: int = 64,
    win_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_fft: int = 512,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
    pre_emph: float = 0.97,
    use_db: bool = True,          # True => 10*log10; False => ln
    cmvn: bool = True,            # normalización por utterance
    plot: bool = False,           # grafica el log-Mel
    title: Optional[str] = None,
    save_pt: bool = True,         # <<< guardar .pt automáticamente
    out_root: str = "tensores",   # raíz para los .pt (paralela a 'audios/')
) -> torch.Tensor:
    """
    Convierte un WAV a espectrograma log-Mel normalizado y
    (opcionalmente) guarda el tensor .pt en tensores/CLASE/archivo.pt.

    @return: feat [frames, n_mels] (torch.FloatTensor)
    """
    # 1) Cargar audio
    wav, sr = _safe_load_audio(wav_path, sr_target)  # [1, N] float32

    # 2) Pre-énfasis
    if pre_emph and pre_emph > 0:
        x = wav[0]
        x = torch.cat([x[:1], x[1:] - pre_emph * x[:-1]], dim=0).unsqueeze(0)
        wav = x

    # 3) Parámetros STFT/Mel
    win_length = int(sr * (win_ms / 1000.0))   # 400 a 16 kHz
    hop_length = int(sr * (hop_ms / 1000.0))   # 160 a 16 kHz
    if fmax is None:
        fmax = sr / 2.0

    # 4) MelSpectrogram (potencia)
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=fmin, f_max=fmax,
        n_mels=n_mels,
        window_fn=torch.hann_window,
        power=2.0,                # espectro de potencia
        norm="slaney",
        mel_scale="htk",
    )(wav)                         # [1, n_mels, frames]

    # 5) Compresión log
    eps = 1e-10
    logmel = 10.0 * torch.log10(melspec + eps) if use_db else torch.log(melspec + eps)

    # 6) Normalización por utterance
    if cmvn:
        mean = logmel.mean(dim=2, keepdim=True)
        std = logmel.std(dim=2, keepdim=True).clamp_min(1e-8)
        logmel = (logmel - mean) / std

    # 7) Salida [frames, n_mels]
    feat = logmel.squeeze(0).transpose(0, 1)   # [T, F]

    # 8) Gráfico opcional
    if plot:
        plt.figure(figsize=(9, 3))
        plt.imshow(feat.numpy().T, aspect="auto", origin="lower", interpolation="nearest")
        plt.xlabel(f"Frames (hop ≈ {hop_ms:.0f} ms)")
        plt.ylabel(f"Bandas Mel ({n_mels})")
        plt.title(title or f"log-Mel ({'dB' if use_db else 'ln'}) – {sr} Hz")
        plt.colorbar(label="amplitud (norm.)" if cmvn else ("dB" if use_db else "log"))
        plt.tight_layout()
        plt.show()

    # 9) Guardar .pt en tensores/CLASE/*.pt (árbol paralelo a audios/)
    if save_pt:
        wav_p = Path(wav_path)
        clase = wav_p.parent.name.upper()
        label = LABEL2ID.get(clase, -1)

        out_dir = Path(out_root) / clase
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pt = out_dir / f"{wav_p.stem}.pt"

        pack = {
            "feat": feat,                 # [T, F]
            "label": label,               # entero de clase
            "meta": {
                "wav_src": str(wav_p),
                "sr": sr,
                "n_mels": n_mels,
                "use_db": use_db,
                "cmvn": cmvn,
            },
        }
        torch.save(pack, out_pt)
        print(f"✔ Guardado {out_pt}")

    return feat  # [frames, n_mels]


# ---------------------------------------------------------------------
# Utilidad opcional: procesar toda la carpeta de audios -> tensores/
# ---------------------------------------------------------------------
def generate_cache(
    wav_root: str = "audios",
    out_root: str = "tensores",
    sr: int = 16000,
    n_mels: int = 64,
    use_db: bool = True,
    cmvn: bool = True,
) -> None:
    """
    Recorre audios/CLASE/*.wav y genera tensores/CLASE/*.pt
    usando los mismos parámetros que wav_to_logmel_torch.
    """
    wav_root_p = Path(wav_root)
    for clase in CLASSES:
        in_dir = wav_root_p / clase
        if not in_dir.exists():
            continue
        for wav in in_dir.rglob("*.wav"):
            wav_to_logmel_torch(
                str(wav),
                sr_target=sr,
                n_mels=n_mels,
                use_db=use_db,
                cmvn=cmvn,
                plot=False,
                save_pt=True,
                out_root=out_root,
            )
