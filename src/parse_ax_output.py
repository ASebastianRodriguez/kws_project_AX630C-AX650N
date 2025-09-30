#!/usr/bin/env python3
import argparse
import os
import sys
import struct
import math
from typing import List

def read_logits(path: str, expected_floats: int = 8) -> List[float]:
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != expected_floats * 4:
        raise ValueError(f"File size {len(data)} bytes isn't {expected_floats*4} (expected {expected_floats} float32s).")
    # Little-endian float32 (standard on most systems)
    floats = list(struct.unpack("<" + "f"*expected_floats, data))
    return floats

def softmax(xs: List[float]) -> List[float]:
    # numerically stable softmax
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def normalize_probs(xs: List[float]) -> List[float]:
    s = sum(xs)
    if s == 0:
        return [0.0]*len(xs)
    return [max(0.0, min(1.0, x / s)) for x in xs]

def main():
    parser = argparse.ArgumentParser(description="Parse AX model output.bin (8 float32 values) and print probabilities.")
    parser.add_argument("bin_path", help="Path to output.bin (32 bytes, 8 float32).")
    parser.add_argument("--assume-probs", action="store_true", help="Treat input values as probabilities (skip softmax and just normalize). Default: treat as logits and apply softmax.")
    parser.add_argument("--classes", nargs="*", default=["ACOPLAR","CANCELAR","CONTINUAR","FONDO","LEVANTADO","PRINCIPAL","REPETIR","SALIR"], help="Optional class names (default: 8 KWS classes).")
    parser.add_argument("--precision", type=int, default=6, help="Decimal places for printing (default: 6).")
    args = parser.parse_args()

    try:
        vals = read_logits(args.bin_path, expected_floats=len(args.classes))
    except Exception as e:
        print(f"Error reading {args.bin_path}: {e}", file=sys.stderr)
        sys.exit(1)

    if args.assume_probs:
        probs = normalize_probs(vals)
    else:
        probs = softmax(vals)

    # Identify top class
    top_idx = max(range(len(probs)), key=lambda i: probs[i])
    top_cls = args.classes[top_idx] if top_idx < len(args.classes) else str(top_idx)

    # Pretty print
    print(f"\nFile: {args.bin_path}")
    print(f"Length: {len(vals)} values (float32)")
    print("\nProbabilities:")
    width = max(len(c) for c in args.classes)
    for i, (c, p) in enumerate(zip(args.classes, probs)):
        print(f"  [{i}] {c.ljust(width)} : {p:.{args.precision}f}")
    print(f"\nSum: {sum(probs):.{args.precision}f}")
    print(f"Top-1: [{top_idx}] {top_cls} ({probs[top_idx]*100:.2f}%)")

if __name__ == "__main__":
    main()
