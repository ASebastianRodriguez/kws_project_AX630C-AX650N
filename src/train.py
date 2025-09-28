from __future__ import annotations
import argparse, time, os
import torch, torch.nn as nn, torch.optim as optim
from dataset_from_cache import make_loaders, CLASSES
from model_crnn import CRNNSmall
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


# ---------- matriz de confusión ----------
def confusion_matrix(model, loader, device, ncls=8):
    cm = np.zeros((ncls, ncls), dtype=int)
    model.eval()
    with torch.no_grad():
        for feats, labels, _ in loader:
            feats = feats.unsqueeze(1).to(device)   # [B,1,T,F]
            preds = model(feats).argmax(1).cpu().numpy()
            labs  = labels.numpy()
            for y, yhat in zip(labs, preds):
                cm[y, yhat] += 1
    return cm
# ------------------------------------------------


def train_main(
    cache_root: str,
    epochs: int = 20,
    bs: int = 32,
    lr: float = 1e-3,
    ckpt_dir: str = "/data/trained_tensor",
    ckpt_name: str = "kws_crnn.pt",
    n_mels: int = 64,
    early_stop: bool = False,
    patience: int = 5,
    min_delta: float = 1e-4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # Ruta final del checkpoint
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    # loaders
    train_loader, val_loader, test_loader = make_loaders(cache_root, bs)

    # modelo
    model = CRNNSmall(n_mels=n_mels).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_ep = 0
    wait = 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        # ---------- train ----------
        model.train()
        tr_loss, tr_acc = 0.0, 0
        for feats, labels, _ in train_loader:
            feats, labels = feats.unsqueeze(1).to(device), labels.to(device)
            opt.zero_grad()
            out = model(feats)
            loss = crit(out, labels)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(labels)
            tr_acc  += (out.argmax(1) == labels).sum().item()
        tr_loss /= len(train_loader.dataset)
        tr_acc  /= len(train_loader.dataset)

        # ---------- val ----------
        model.eval()
        val_loss, val_acc = 0.0, 0
        with torch.no_grad():
            for feats, labels, _ in val_loader:
                feats, labels = feats.unsqueeze(1).to(device), labels.to(device)
                out = model(feats)
                loss = crit(out, labels)
                val_loss += loss.item() * len(labels)
                val_acc  += (out.argmax(1) == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc  /= len(val_loader.dataset)

        print(f"Epoch {ep:02d} | train {tr_loss:.4f}/{tr_acc:.3f} | val {val_loss:.4f}/{val_acc:.3f}")

        # ---------- early stopping / best checkpoint ----------
        improved = (best_val - val_loss) > min_delta
        if improved:
            best_val, best_ep = val_loss, ep
            wait = 0
            torch.save(model.state_dict(), ckpt_path)  # guardamos el mejor
        else:
            wait += 1
            if early_stop and wait >= patience:
                print(f"Early stop en epoch {ep} (mejor val {best_val:.4f} en {best_ep})")
                break

    # ---------- test ----------
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    tst_loss, tst_acc = 0.0, 0
    with torch.no_grad():
        for feats, labels, _ in test_loader:
            feats, labels = feats.unsqueeze(1).to(device), labels.to(device)
            out = model(feats)
            loss = crit(out, labels)
            tst_loss += loss.item() * len(labels)
            tst_acc  += (out.argmax(1) == labels).sum().item()
    tst_loss /= len(test_loader.dataset)
    tst_acc  /= len(test_loader.dataset)

    print(f"Test   | loss {tst_loss:.4f} | acc {tst_acc:.3f}")
    print(f"Checkpoint guardado en: {ckpt_path} | tiempo total: {time.time()-t0:.1f}")

    # ---------- matriz de confusión ----------
    cm = confusion_matrix(model, test_loader, device, ncls=len(CLASSES))
    print("Confusion matrix:\n", cm)

    # ---------- visualización con heatmap ----------
    plt.figure(figsize=(8, 6))
    if _HAS_SNS:
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES
        )
    else:
        plt.imshow(cm, aspect="auto", interpolation="nearest")
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de confusión - KWS")
    plt.tight_layout()

    png_path = ckpt_path.replace(".pt", "_confusion.png")
    plt.savefig(png_path, dpi=150)
    print(f"Matriz de confusión guardada en: {png_path}")
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt-dir", type=str, default="/data/trained_tensor", help="Directorio donde se guardará el checkpoint")
    ap.add_argument("--ckpt-name", type=str, default="kws_crnn.pt", help="Nombre de archivo de checkpoint")
    ap.add_argument("--n-mels", type=int, default=64, help="Número de bandas Mel (default: 64)")
    # Early stopping opcional
    ap.add_argument("--early-stop", action="store_true", help="Activa early stopping basado en val loss")
    ap.add_argument("--patience", type=int, default=5, help="Epochs sin mejora antes de frenar (default: 5)")
    ap.add_argument("--min-delta", type=float, default=1e-4, help="Mejora mínima en val loss para resetear paciencia (default: 1e-4)")
    args = ap.parse_args()

    train_main(
        cache_root=args.cache_root,
        epochs=args.epochs,
        bs=args.batch_size,
        lr=args.lr,
        ckpt_dir=args.ckpt_dir,
        ckpt_name=args.ckpt_name,
        n_mels=args.n_mels,
        early_stop=args.early_stop,
        patience=args.patience,
        min_delta=args.min_delta,
    )
