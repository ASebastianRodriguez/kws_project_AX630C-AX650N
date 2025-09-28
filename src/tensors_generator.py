# main.py
from __future__ import annotations
import argparse
from pathlib import Path
from features import wav_to_logmel_torch, generate_cache  # generate_cache queda por si querés usarla aparte

def run_batch(wav_root: str, out_root: str, sr: int, n_mels: int, use_db: bool, cmvn: bool, every: int) -> None:
    """
    Recorre wav_root buscando *.wav y genera los .pt en out_root replicando el árbol de clases.
    Muestra progreso cada 'every' archivos.
    """
    wav_root_p = Path(wav_root)
    if not wav_root_p.exists():
        raise FileNotFoundError(f"No existe wav_root: {wav_root}")

    wavs = sorted([p for p in wav_root_p.rglob("*.wav") if p.is_file()])
    if not wavs:
        print(f"Sin .wav en {wav_root}")
        return

    total = len(wavs)
    print(f"Encontrados {total} archivos .wav en '{wav_root}'. Generando tensores en '{out_root}' ...")

    done = 0
    for wav in wavs:
        # Llama a tu función que además guarda el .pt en out_root/<CLASE>/<stem>.pt
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
        done += 1
        if every > 0 and (done % every == 0 or done == total):
            print(f"[{done}/{total}] procesados")

    print(f"Listo: {done}/{total} archivos procesados.")

def main() -> None:
    parser = argparse.ArgumentParser(description="WAV → log-Mel (PyTorch/torchaudio)")

    # modo normal (un archivo)
    parser.add_argument("wav_path", nargs="?", type=str, help="Ruta al archivo WAV (mono)")
    parser.add_argument("--sr", type=int, default=16000, help="Frecuencia destino")
    parser.add_argument("--n_mels", type=int, default=64, help="Bandas Mel")
    parser.add_argument("--db", action="store_true", help="Usar dB (10*log10) en lugar de ln")
    parser.add_argument("--no-cmvn", action="store_true", help="Desactivar normalización por utterance")
    parser.add_argument("--plot", action="store_true", help="Graficar el log-Mel")
    parser.add_argument("--save-pt", action="store_true", help="Generar y guardar el .pt (árbol de clases)")
    parser.add_argument("--out-root", type=str, default="tensores", help="Raíz para los .pt (paralela a './audios/')")

    # modo batch
    parser.add_argument("--generate-cache", action="store_true",
                        help="Procesa todos los .wav en --wav-root y genera .pt en --out-root")
    parser.add_argument("--wav-root", type=str, default="audios", help="Raíz que contiene las carpetas por clase con .wav")
    parser.add_argument("--every", type=int, default=50, help="Muestra progreso cada N archivos en modo batch")

    args = parser.parse_args()

    # modo batch: ignora wav_path y procesa todo con progreso
    if args.generate_cache:
        run_batch(
            wav_root=args.wav_root,
            out_root=args.out_root,
            sr=args.sr,
            n_mels=args.n_mels,
            use_db=args.db,
            cmvn=not args.no_cmvn,
            every=args.every,
        )
        return

    # modo individual
    if not args.wav_path:
        parser.error("Debe especificar wav_path o usar --generate-cache")

    wav_path = Path(args.wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"No se encontró: {wav_path}")

    feat = wav_to_logmel_torch(
        str(wav_path),
        sr_target=args.sr,
        n_mels=args.n_mels,
        use_db=args.db,
        cmvn=not args.no_cmvn,
        plot=args.plot,
        title=f"log-Mel de {wav_path.name}",
        save_pt=args.save_pt,
        out_root=args.out_root
    )
    print(f"OK · forma de las características: {tuple(feat.shape)}  (frames, n_mels)")

if __name__ == "__main__":
    main()
