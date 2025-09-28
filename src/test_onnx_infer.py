# test_onnx_infer.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort

# Ajustá las clases si tu orden es distinto
CLASSES = ["ACOPLAR","CANCELAR","CONTINUAR","FONDO",
           "LEVANTADO","PRINCIPAL","REPETIR","SALIR"]

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def load_session(onnx_path: str) -> ort.InferenceSession:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # Chequeo de I/O
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    if "features" not in in_names:
        raise RuntimeError(f"Entrada 'features' no encontrada. Entradas: {in_names}")
    if len(out_names) != 1:
        raise RuntimeError(f"Se esperaba 1 salida (logits). Salidas: {out_names}")
    return sess

def infer_dummy(sess: ort.InferenceSession, T: int, n_mels: int, random: bool) -> None:
    x = np.random.randn(1, 1, T, n_mels).astype("float32") if random \
        else np.zeros((1, 1, T, n_mels), dtype="float32")
    (logits,) = sess.run(None, {"features": x})
    probs = softmax_np(logits)[0]
    pred = int(np.argmax(probs))
    print(f"[DUMMY] shape logits: {logits.shape}  | pred: {CLASSES[pred]}")
    for i, p in enumerate(probs):
        print(f"  {CLASSES[i]:<10} {p:7.4f}")

def infer_from_pt(sess: ort.InferenceSession, pt_path: str) -> None:
    pack = torch.load(pt_path, map_location="cpu")
    feat = pack["feat"].float()  # [T, F]
    T, Fm = feat.shape
    x = feat.unsqueeze(0).unsqueeze(0).numpy()  # [1,1,T,F]
    (logits,) = sess.run(None, {"features": x})
    probs = softmax_np(logits)[0]
    pred = int(np.argmax(probs))

    meta = pack.get("meta", {})
    label = int(pack.get("label", -1))
    expected = CLASSES[label] if 0 <= label < len(CLASSES) else "desconocida"

    print(f"[PT]  archivo: {pt_path}")
    print(f"     feat: [T,F]=[{T},{Fm}]  | meta: {meta}")
    print(f"     pred: {CLASSES[pred]}  (label guardado: {expected})")
    for i, p in enumerate(probs):
        print(f"  {CLASSES[i]:<10} {p:7.4f}")

def main():
    ap = argparse.ArgumentParser(description="Prueba de inferencia ONNX (KWS CRNN)")
    ap.add_argument("--onnx", type=str, default="kws_es_crnn.onnx", help="Ruta al modelo ONNX")
    sub = ap.add_subparsers(dest="mode", required=True)

    sp_dummy = sub.add_parser("dummy", help="Inferencia con entrada sintética")
    sp_dummy.add_argument("--T", type=int, default=98, help="Frames temporales (ej. 98 ~ 1s @ 10 ms)")
    sp_dummy.add_argument("--n-mels", type=int, default=64, help="Número de bandas Mel")
    sp_dummy.add_argument("--random", action="store_true", help="Usar ruido aleatorio en vez de ceros")

    sp_pt = sub.add_parser("pt", help="Inferencia desde un archivo .pt del cache")
    sp_pt.add_argument("pt_path", type=str, help="Ruta a tensores/CLASE/archivo.pt")

    args = ap.parse_args()
    if not Path(args.onnx).exists():
        raise FileNotFoundError(f"No se encontró ONNX: {args.onnx}")

    sess = load_session(args.onnx)

    if args.mode == "dummy":
        infer_dummy(sess, T=args.T, n_mels=args["n_mels"] if isinstance(args, dict) else args.n_mels, random=args.random)
    else:
        infer_from_pt(sess, args.pt_path)

if __name__ == "__main__":
    main()
