#!/usr/bin/env python3
"""
Generate test inputs for ax_run_model from samples produced by make_calib.py.

This script ingests any of the outputs that make_calib.py can emit:
  - Numpy:         input_XXXXX.npy            (np.ndarray float32, shape (1,1,T,F) or (T,F))
  - Binary:        input_XXXXX.bin            (raw float32, expected layout (1,1,T,F))
  - NumpyObject:   input_XXXXX.npy            (dict{"input": np.ndarray})
  - Image (PNG):   input_XXXXX.png            (grayscale visualization; lossy — optional)
  - TAR archive:   calib_kws.tar              (flat list of the above)

It converts them into ax_run_model-friendly folder layout:
  <out_dir>/input/0/input.bin
  <out_dir>/input/1/input.bin
  ...
And writes a list file:
  <out_dir>/list.txt                        (contains: 0, 1, 2, ...)
Also prepares an empty output directory:
  <out_dir>/output/

By default, this script expects the model input tensor to be named 'input'.
Use --tensor-name to change the filename accordingly when using --use-tensor-name in ax_run_model.

Example usage:
  python generate_test_files.py \
      --src /data/calib_samples/input \
      --out-dir /root/kws_run \
      --t 98 --n-mels 64 --limit 50

  # If you have a TAR from make_calib.py:
  python generate_test_files.py --src /data/calib_samples/calib_kws.tar --out-dir /root/kws_run

Then run inference on device:
  ax_run_model -m /root/kws_int8.axmodel \
      -i /root/kws_run/input -o /root/kws_run/output -l /root/kws_run/list.txt \
      --use-tensor-name

Notes:
- PNG inputs created by make_calib.py are normalized visualizations (1–99 percentiles) and are NOT strictly invertible to the original features. This script only converts them on best-effort if you pass --allow-image.
- For Binary inputs, we assume float32, contiguous, with expected element count (1*1*T*F). The raw layout used by make_calib.py matches ax_run_model.
"""
from __future__ import annotations
import argparse
import io
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

SUPPORTED_EXTS = {".npy", ".bin", ".png"}


def _discover_files(src: Path) -> List[Path]:
    """Discover candidate input_* files under src.
    Accepts either a directory containing files, or a single file (like .tar).
    """
    if src.is_file():
        return [src]
    files = []
    for ext in ("*.npy", "*.bin", "*.png", "*.tar", "*.tar.gz"):
        files.extend(sorted(src.rglob(ext)))
    return files


def _extract_tar_to_tmp(tar_path: Path) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="gen_ax_inputs_"))
    mode = "r:gz" if str(tar_path).endswith(".tar.gz") else "r"
    with tarfile.open(tar_path, mode) as tar:
        # The make_calib.py TAR stores flat files (no subdirs)
        tar.extractall(tmpdir)
    return tmpdir


def _ensure_tensor(x: np.ndarray, T: int, F: int) -> np.ndarray:
    """Coerce array to float32 and shape (1,1,T,F). Accepts (T,F) or (1,1,T,F)."""
    if x.ndim == 2 and x.shape == (T, F):
        x = x[None, None, :, :]
    elif x.ndim == 4 and x.shape[2:] == (T, F):
        # assume (N,C,T,F); keep only first N,C for single sample
        if x.shape[0] != 1 or x.shape[1] != 1:
            x = x[:1, :1, :, :]
    else:
        raise ValueError(f"Unexpected tensor shape {x.shape}; expected (T,F) or (1,1,T,F) with T={T}, F={F}")
    return x.astype(np.float32, copy=False)


def _load_npy(path: Path, T: int, F: int) -> np.ndarray:
    # Can be an ndarray or a dict/object (when saved as NumpyObject)
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray):
        return _ensure_tensor(arr, T, F)
    if isinstance(arr, dict) and "input" in arr:
        return _ensure_tensor(arr["input"], T, F)
    # Some pickled objects may load as 0-d array containing dict
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        obj = arr.item()
        if isinstance(obj, dict) and "input" in obj:
            return _ensure_tensor(obj["input"], T, F)
    raise ValueError(f"Unsupported .npy content in {path}")


def _load_png(path: Path, T: int, F: int, assume_lossy: bool) -> np.ndarray:
    if not _HAS_PIL:
        raise RuntimeError("PNG requires Pillow; install 'Pillow' or convert from .npy/ .bin instead.")
    if not assume_lossy:
        raise RuntimeError("PNG from make_calib.py is a lossy visualization; re-run with --allow-image to accept it.")
    img = Image.open(path).convert("L")  # grayscale
    arr = np.asarray(img, dtype=np.float32) / 255.0  # 0..1 scaled (best effort)
    # Ensure (T,F)
    if arr.shape != (T, F):
        arr = _resize_or_pad_2d(arr, T, F)
    return arr[None, None, :, :].astype(np.float32)


def _resize_or_pad_2d(arr: np.ndarray, T: int, F: int) -> np.ndarray:
    # Simple best-effort: crop or pad with edge values to reach (T,F)
    t, f = arr.shape[:2]
    # crop
    arr = arr[:T, :F]
    # pad if needed
    if arr.shape[0] < T:
        pad_t = np.tile(arr[-1:, :], (T - arr.shape[0], 1)) if arr.size else np.zeros((T - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad_t], axis=0)
    if arr.shape[1] < F:
        pad_f = np.tile(arr[:, -1:], (1, F - arr.shape[1])) if arr.size else np.zeros((arr.shape[0], F - arr.shape[1]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad_f], axis=1)
    return arr


def _copy_or_validate_bin(src: Path, dst: Path, T: int, F: int, strict: bool) -> None:
    # Make_calib Binary is raw float32 (1,1,T,F). We copy as-is.
    # Optional strict mode can validate file size equals T*F*4*1*1 bytes.
    if strict:
        expected_bytes = T * F * 4  # float32, (1,1,T,F) contiguous equals T*F elements
        size = src.stat().st_size
        if size != expected_bytes:
            raise ValueError(f"Unexpected .bin size: {size} bytes (expected {expected_bytes}) for {src}")
    shutil.copy2(src, dst)


def _iter_make_calib_files(root: Path) -> Iterable[Path]:
    # Prefer deterministic order
    for ext in (".npy", ".bin", ".png"):
        for p in sorted(root.glob(f"input_*{ext}")):
            yield p


def _prepare_out_dirs(out_dir: Path) -> Tuple[Path, Path, Path]:
    in_dir = out_dir / "input"
    out_pred = out_dir / "output"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_pred.mkdir(parents=True, exist_ok=True)
    return in_dir, out_pred, out_dir / "list.txt"


def convert_samples(src: Path, out_dir: Path, T: int, F: int, limit: int, tensor_name: str,
                    allow_image: bool, strict_bin: bool) -> int:
    # Resolve source: if TAR, extract to temp; if directory, process directly
    cleanup_dir: Path | None = None
    if src.is_file() and src.suffix in {".tar", ".gz"} or str(src).endswith((".tar", ".tar.gz")):
        cleanup_dir = _extract_tar_to_tmp(src)
        work_root = cleanup_dir
    else:
        # If src points to the parent, but there is an 'input/' inside, use it.
        work_root = src / "input" if (src.is_dir() and (src / "input").exists()) else src

    in_dir, out_pred, list_file = _prepare_out_dirs(out_dir)
    produced = 0

    for idx, p in enumerate(_iter_make_calib_files(work_root)):
        if limit and produced >= limit:
            break
        sample_dir = in_dir / str(produced)
        sample_dir.mkdir(parents=True, exist_ok=True)
        dst = sample_dir / f"{tensor_name}.bin"

        try:
            if p.suffix == ".npy":
                arr = _load_npy(p, T, F)
                arr.tofile(dst)
            elif p.suffix == ".bin":
                _copy_or_validate_bin(p, dst, T, F, strict=strict_bin)
            elif p.suffix == ".png":
                arr = _load_png(p, T, F, assume_lossy=allow_image)
                arr.tofile(dst)
            else:
                print(f"[SKIP] Unsupported extension: {p}")
                produced -= 1  # neutralize increment below
                continue
        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")
            # remove partially created dir
            try:
                if dst.exists():
                    dst.unlink()
                if sample_dir.exists() and not any(sample_dir.iterdir()):
                    sample_dir.rmdir()
            except Exception:
                pass
            continue

        produced += 1

    # Write list.txt
    with open(list_file, "w", encoding="utf-8") as f:
        for i in range(produced):
            f.write(f"{i}\n")

    # Optional: write a convenience run command
    with open(out_dir / "RUN_AX.sh", "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\n")
        f.write("# Example: adjust model path if needed\n")
        f.write("ax_run_model -m /root/kws_int8.axmodel \\\n -i {}/input -o {}/output -l {}/list.txt \\\n --use-tensor-name\n".format(out_dir, out_dir, out_dir))
    os.chmod(out_dir / "RUN_AX.sh", 0o755)

    # Cleanup temp extraction if used
    if cleanup_dir is not None:
        shutil.rmtree(cleanup_dir, ignore_errors=True)

    return produced


def main():
    ap = argparse.ArgumentParser(description="Convert make_calib.py samples to ax_run_model inputs")
    ap.add_argument("--src", type=Path, required=True,
                    help="Directory containing input_* files OR a calib_kws.tar produced by make_calib.py")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Destination root for ax_run_model inputs (will create input/, output/, list.txt)")
    ap.add_argument("--t", type=int, default=98, help="Frames T expected by model")
    ap.add_argument("--n-mels", type=int, default=64, help="Mel bands F expected by model")
    ap.add_argument("--limit", type=int, default=0, help="Max number of samples to convert (0 = all)")
    ap.add_argument("--tensor-name", type=str, default="input", help="Input tensor name (affects file name: <tensor>.bin)")
    ap.add_argument("--allow-image", action="store_true", help="Allow converting PNG (lossy visualization) to features")
    ap.add_argument("--strict-bin", action="store_true", help="Validate .bin size equals T*F*4 bytes (float32)")

    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    count = convert_samples(
        src=args.src,
        out_dir=out_dir,
        T=args.t,
        F=args.n_mels,
        limit=args.limit,
        tensor_name=args.tensor_name,
        allow_image=args.allow_image,
        strict_bin=args.strict_bin,
    )

    print(f"Done. Generated {count} sample(s) under {out_dir}/input and wrote {out_dir}/list.txt")
    print("Run: ax_run_model -m /root/kws_int8.axmodel -i {0}/input -o {0}/output -l {0}/list.txt --use-tensor-name".format(out_dir))


if __name__ == "__main__":
    main()
