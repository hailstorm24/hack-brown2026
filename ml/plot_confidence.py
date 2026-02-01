"""Plot model confidence (predicted prob) with points colored by actual label (pos vs neg)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

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
VAL_RATIO = 0.15
SEED = 42


def collect_confidence_and_labels(
    dataset: FlightWindowDataset,
    model: RelevanceModel,
    device: torch.device,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on val split; return (confidence, labels). confidence = sigmoid(logits/temperature)."""
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    _, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    all_conf, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            x, y, c = [t.to(device) for t in batch]
            logits = model(x, c)
            conf = torch.sigmoid(logits / temperature).cpu().numpy()
            all_conf.append(conf)
            all_y.append(y.cpu().numpy())
    confidence = np.concatenate(all_conf)
    labels = np.concatenate(all_y)
    return confidence, labels


def main():
    parser = argparse.ArgumentParser(description="Plot confidence (pos vs neg)")
    parser.add_argument("--model", type=str, default=str(DATA_DIR / "model.pt"))
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "confidence_plot.png"))
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--temperature", type=float, default=1.0, help="Softens probs: sigmoid(logits/T); T>1 spreads points")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib: pip install matplotlib")
        sys.exit(1)

    dataset = FlightWindowDataset()
    if len(dataset) == 0:
        print("No dataset. Run training first.")
        sys.exit(1)

    model = RelevanceModel().to(DEVICE)
    path = Path(args.model)
    if not path.exists():
        print(f"Model not found: {path}. Run training first.")
        sys.exit(1)
    ckpt = torch.load(path, weights_only=True, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    confidence, labels = collect_confidence_and_labels(
        dataset, model, DEVICE, val_ratio=args.val_ratio, seed=args.seed, temperature=args.temperature
    )
    confidence = confidence * 2 - 1  # scale [0, 1] -> [-1, 1]
    neg = labels == 0
    pos = labels == 1
    # Jitter x so points don't stack: x = 0 or 1 + small noise
    rng = np.random.default_rng(args.seed)
    jitter = 0.08
    x_neg = 0 + rng.uniform(-jitter, jitter, neg.sum())
    x_pos = 1 + rng.uniform(-jitter, jitter, pos.sum())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x_neg, confidence[neg], c="tab:blue", alpha=0.6, s=14, label="No Corresponding Event")
    ax.scatter(x_pos, confidence[pos], c="tab:orange", alpha=0.8, s=24, label="Corresponding Event")
    ax.set_xlabel("Relevance")
    ax.set_ylabel("Confidence")
    ax.set_ylim(-1, 1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Corresponding Event", "Corresponding Event"])
    ax.set_xlim(-0.35, 1.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True)
    ax.set_title("Confidence vs Relevance")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
