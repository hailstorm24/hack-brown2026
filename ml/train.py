"""Train RelevanceModel with weighted BCE; optional val split and metrics."""
import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split, Subset
    from sklearn.model_selection import StratifiedKFold, KFold
except ImportError as e:
    print("Missing dependency:", e)
    print("Install with: pip install -r ml/requirements.txt")
    print("If using a venv: source .venv/bin/activate  then  pip install -r ml/requirements.txt")
    sys.exit(1)

# Allow running as script: python ml/train.py from repo root
if __package__ is None:
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from ml.config import DATA_DIR
    from ml.data import FlightWindowDataset
    from ml.model import RelevanceModel
else:
    from .config import DATA_DIR
    from .data import FlightWindowDataset
    from .model import RelevanceModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 0.02  # light L2 to reduce overconfident positives
VAL_RATIO = 0.15
SEED = 42


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def weighted_bce(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    """BCEWithLogitsLoss with pos_weight for class imbalance."""
    return nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pos_weight)


def validate_data(dataset: FlightWindowDataset, model: RelevanceModel, device: torch.device) -> None:
    """Load one batch, run forward pass; raise on error so we fail fast."""
    if len(dataset) == 0:
        return
    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    x, y, c = [t.to(device) for t in batch]
    with torch.no_grad():
        _ = model(x, c)
    print("Validate: one batch OK (no errors).")


def confusion_matrix_from_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[int, int, int, int]:
    """Run model on loader; return (TN, FP, FN, TP)."""
    model.eval()
    all_pred, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            x, y, c = [t.to(device) for t in batch]
            logits = model(x, c)
            pred = (logits > 0).long()
            all_pred.append(pred.cpu().numpy())
            all_y.append(y.cpu().numpy())
    pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_y)
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tp = int(((pred == 1) & (y_true == 1)).sum())
    return tn, fp, fn, tp


def print_confusion_matrix(tn: int, fp: int, fn: int, tp: int, title: str = "Confusion matrix") -> None:
    """Print 2x2 matrix (FP vs FN) and summary."""
    print(f"\n--- {title} ---")
    print("                 Predicted Neg   Predicted Pos")
    print(f"Actual Neg           {tn:6} (TN)      {fp:6} (FP)")
    print(f"Actual Pos           {fn:6} (FN)      {tp:6} (TP)")
    print(f"\nFalse Positives: {fp}   False Negatives: {fn}")
    total = tn + fp + fn + tp
    if total:
        acc = (tn + tp) / total
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        print(f"Accuracy: {acc:.4f}   Recall: {recall:.4f}   Precision: {precision:.4f}")
    print()


def _save_matrix(path: str, tn: int, fp: int, fn: int, tp: int) -> None:
    """Write confusion matrix to CSV: TN, FP, FN, TP."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("TN,FP,FN,TP\n")
        f.write(f"{tn},{fp},{fn},{tp}\n")
    print(f"Confusion matrix saved to {path}")


def run_fold(
    dataset: FlightWindowDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> tuple[float, float, torch.nn.Module]:
    """Train one fold; return (val_loss, val_acc, model)."""
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = RelevanceModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            x, y, c = [t.to(device) for t in batch]
            opt.zero_grad()
            logits = model(x, c)
            loss = weighted_bce(logits, y, pos_weight)
            loss.backward()
            opt.step()
        model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            x, y, c = [t.to(device) for t in batch]
            logits = model(x, c)
            val_loss += weighted_bce(logits, y, pos_weight).item()
            pred = (logits > 0).long()
            correct += (pred == y).sum().item()
            total += y.size(0)
    val_loss /= len(val_loader) if val_loader else 1
    acc = correct / total if total else 0
    return float(val_loss), float(acc), model


def main():
    parser = argparse.ArgumentParser(description="Train M&A relevance model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="L2 regularization")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO, help="Used only if --folds 0")
    parser.add_argument("--folds", type=int, default=5, help="K-fold; use 0 for single train/val split")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--save", type=str, default=str(DATA_DIR / "model.pt"))
    parser.add_argument("--no-cache", action="store_true", help="Rebuild windows and embeddings cache")
    parser.add_argument("--fn-weight", type=float, default=5.0, help="Penalize false negatives more than FPs; 1.0 = balance only")
    parser.add_argument("--matrix-out", type=str, default="", help="Save confusion matrix to CSV (TN,FP,FN,TP)")
    args = parser.parse_args()
    set_seed(args.seed)

    dataset = FlightWindowDataset()
    if len(dataset) == 0:
        print("No windows built (empty flights or windows). Exiting.")
        return
    n_pos = int(dataset.windows_df["label"].sum()) if not dataset.windows_df.empty else 0
    n_pos = max(1, n_pos)
    n_neg = max(1, len(dataset) - n_pos)
    pos_weight = torch.tensor([args.fn_weight * (n_neg / n_pos)], dtype=torch.float32, device=DEVICE)
    print(f"fn_weight={args.fn_weight}  n_pos={n_pos}  n_neg={n_neg}  pos_weight={pos_weight.item():.2f}")
    model = RelevanceModel().to(DEVICE)
    print("Validating data and model (one batch)...")
    try:
        validate_data(dataset, model, DEVICE)
    except Exception as e:
        print("Validation failed:", e)
        raise

    if args.folds <= 0:
        n_val = max(1, int(len(dataset) * args.val_ratio))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                x, y, c = [t.to(DEVICE) for t in batch]
                opt.zero_grad()
                logits = model(x, c)
                loss = weighted_bce(logits, y, pos_weight)
                loss.backward()
                opt.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y, c = [t.to(DEVICE) for t in batch]
                    logits = model(x, c)
                    val_loss += weighted_bce(logits, y, pos_weight).item()
                    pred = (logits > 0).long()
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            val_loss /= len(val_loader) if val_loader else 1
            acc = correct / total if total else 0
            print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={acc:.4f}")
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "pos_weight": pos_weight.cpu()}, args.save)
        print(f"Saved to {args.save}")
        tn, fp, fn, tp = confusion_matrix_from_loader(model, val_loader, DEVICE)
        print_confusion_matrix(tn, fp, fn, tp, "Val set: FP vs FN")
        if args.matrix_out:
            _save_matrix(args.matrix_out, tn, fp, fn, tp)
        return

    # K-fold (stratified if both classes present; else plain KFold)
    labels = dataset.windows_df["label"].values
    try:
        kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = list(kf.split(np.zeros(len(dataset)), labels))
    except ValueError:
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = list(kf.split(np.arange(len(dataset))))
    fold_results = []
    last_model = model
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"--- Fold {fold + 1}/{args.folds} ---")
        vl, acc, fold_model = run_fold(dataset, train_idx, val_idx, args, DEVICE, pos_weight)
        fold_results.append((vl, acc))
        last_model = fold_model
        print(f"Fold {fold + 1}  val_loss={vl:.4f}  val_acc={acc:.4f}")
    mean_loss = np.mean([r[0] for r in fold_results])
    mean_acc = np.mean([r[1] for r in fold_results])
    print(f"K-fold ({args.folds})  mean val_loss={mean_loss:.4f}  mean val_acc={mean_acc:.4f}")
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": last_model.state_dict(), "pos_weight": pos_weight.cpu()}, args.save)
    print(f"Saved to {args.save}")
    # Confusion matrix on last fold's val set
    _, last_val_idx = splits[-1]
    last_val_loader = DataLoader(Subset(dataset, last_val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
    tn, fp, fn, tp = confusion_matrix_from_loader(last_model, last_val_loader, DEVICE)
    print_confusion_matrix(tn, fp, fn, tp, "Last fold val: FP vs FN")
    if args.matrix_out:
        _save_matrix(args.matrix_out, tn, fp, fn, tp)


if __name__ == "__main__":
    main()
