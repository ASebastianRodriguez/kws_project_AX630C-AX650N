# make_calib.py
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import torch
import random
import tarfile

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def _save_numpy(path: Path, arr: np.ndarray) -> None:
    np.save(path, arr)


def _save_binary(path: Path, arr: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(arr.tobytes())


def _save_numpy_object(path: Path, arr: np.ndarray) -> None:
    obj = {"input": arr}
    np.save(path, obj, allow_pickle=True)


def _save_image(path: Path, arr: np.ndarray) -> None:
    if not _HAS_PIL:
        raise RuntimeError("Formato Image requiere Pillow. Instalá 'Pillow' o usá otro formato.")
    img = arr[0, 0]  # (T,F)
    vmin = np.percentile(img, 1.0)
    vmax = np.percentile(img, 99.0)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    img_norm = (np.clip(img, vmin, vmax) - vmin) / (vmax - vmin)
    img_u8 = (img_norm * 255.0).astype(np.uint8)
    Image.fromarray(img_u8, mode="L").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-root", type=str, required=True, help="Raíz de tensores .pt (ej: /data/tensores)")
    ap.add_argument("--out-dir", type=str, required=True, help="Directorio de salida (ej: /data/calib_samples)")
    ap.add_argument("--calib-format", type=str, default="Numpy",
                    choices=["Numpy", "Binary", "NumpyObject", "Image"],
                    help="Formato de datos de calibración para Pulsar2")
    ap.add_argument("--t", type=int, default=98, help="Frames T fijos (debe coincidir con ONNX)")
    ap.add_argument("--n-mels", type=int, default=64, help="Número de bandas Mel esperado")
    ap.add_argument("--max-samples", type=int, default=300, help="Cantidad de muestras a exportar")
    ap.add_argument("--seed", type=int, default=123, help="Semilla para muestreo aleatorio")
    ap.add_argument("--tar-name", type=str, default="calib_kws.tar",
                    help="Nombre del archivo TAR de salida (por defecto: calib_kws.tar)")
    args = ap.parse_args()

    random.seed(args.seed)

    pt_paths = sorted(Path(args.pt_root).rglob("*.pt"))
    if not pt_paths:
        raise SystemExit(f"Sin .pt en {args.pt_root}")

    out_root = Path(args.out_dir)
    out_dir = out_root / "input"
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(pt_paths) > args.max_samples:
        pt_paths = random.sample(pt_paths, args.max_samples)

    fmt = args.calib_format
    if fmt == "Numpy":
        ext = ".npy"
        saver = _save_numpy
    elif fmt == "Binary":
        ext = ".bin"
        saver = _save_binary
    elif fmt == "NumpyObject":
        ext = ".npy"
        saver = _save_numpy_object
    elif fmt == "Image":
        ext = ".png"
        saver = _save_image
    else:
        raise SystemExit(f"Formato no soportado: {fmt}")

    count = 0
    for p in pt_paths:
        pack = torch.load(p, map_location="cpu")
        feat = pack["feat"].float()
        T, Fm = feat.shape
        if Fm != args.n_mels:
            continue

        if T == args.t:
            x = feat.unsqueeze(0).unsqueeze(0)
        elif T > args.t:
            x = feat[:args.t, :].unsqueeze(0).unsqueeze(0)
        else:
            pad = torch.zeros(args.t - T, Fm, dtype=feat.dtype)
            x = torch.cat([feat, pad], dim=0).unsqueeze(0).unsqueeze(0)

        arr = x.numpy().astype("float32")
        saver(out_dir / f"input_{count:05d}{ext}", arr)
        count += 1

    print(f"Listo: exportadas {count} muestras en {out_dir} (formato: {fmt})")

    # Crear archivo TAR
    files = sorted(out_dir.glob(f"input_*{ext}"))
    if files:
        tar_path = out_root / args.tar_name
        print(f"Creando archivo TAR en {tar_path} ...")
        with tarfile.open(tar_path, "w") as tar:
            for fpath in files:
                tar.add(fpath, arcname=fpath.name)  # solo el archivo, sin subcarpetas
        print(f"Archivo TAR creado: {tar_path} con {len(files)} archivos")
    else:
        print("⚠️ No se generaron archivos, no se creó el .tar")


if __name__ == "__main__":
    main()
