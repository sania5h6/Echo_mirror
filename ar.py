"""
EchoMirror - Accuracy Evaluation Module (v2)
Features:
  - Uses the evaluation table in MemoryDB (thread-safe)
  - Adds timestamp, date filtering
  - Classification report + accuracy score
  - Confusion matrix visualization
  - Log individual predictions for ground-truth comparison
"""

import sqlite3
import sys
from datetime import datetime, timedelta
from collections import Counter

try:
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[ar.py] sklearn not found — install with: pip install scikit-learn")

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


DB_PATH = "emotions.db"


def _connect():
    return sqlite3.connect(DB_PATH)


def ensure_table():
    """Create evaluation table if it doesn't exist."""
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS evaluation (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT DEFAULT (datetime('now','localtime')),
            predicted   TEXT,
            actual      TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_evaluation(predicted: str, actual: str):
    """Log a single prediction vs ground truth."""
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        INSERT INTO evaluation (timestamp, predicted, actual)
        VALUES (datetime('now','localtime'), ?, ?)
    """, (predicted.lower(), actual.lower()))
    conn.commit()
    conn.close()


def evaluate_accuracy(days=None):
    """
    Print classification report and accuracy.
    Args:
        days: If set, only evaluate data from the last N days.
    """
    if not HAS_SKLEARN:
        print("[ERROR] sklearn required for evaluation.")
        return

    conn = _connect()
    c = conn.cursor()

    if days:
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        c.execute("SELECT predicted, actual FROM evaluation WHERE timestamp >= ?", (since,))
    else:
        c.execute("SELECT predicted, actual FROM evaluation")

    data = c.fetchall()
    conn.close()

    if not data:
        print("[Evaluation] No evaluation data yet.")
        print("  Tip: Use log_evaluation(predicted, actual) to add ground truth.")
        return

    y_pred = [d[0] for d in data]
    y_true = [d[1] for d in data]

    print(f"\n[EchoMirror Evaluation] {len(data)} samples")
    if days:
        print(f"  Filtered: last {days} days")
    print(f"  Accuracy: {accuracy_score(y_true, y_pred):.2%}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Emotion-level stats
    print("Per-emotion counts:")
    pred_counts = Counter(y_pred)
    true_counts = Counter(y_true)
    all_emotions = sorted(set(y_pred + y_true))
    for emo in all_emotions:
        print(f"  {emo:10s}  predicted: {pred_counts.get(emo, 0):4d}  actual: {true_counts.get(emo, 0):4d}")

    return y_true, y_pred


def plot_confusion_matrix(y_true=None, y_pred=None, days=None):
    """Display a colored confusion matrix."""
    if not HAS_PLOT or not HAS_SKLEARN:
        print("[ERROR] matplotlib and sklearn required.")
        return

    if y_true is None or y_pred is None:
        result = evaluate_accuracy(days)
        if not result:
            return
        y_true, y_pred = result

    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    im = ax.imshow(cm, interpolation="nearest", cmap="YlOrRd")
    ax.figure.colorbar(im, ax=ax, shrink=0.8)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel="True Emotion", xlabel="Predicted Emotion")

    ax.set_title("EchoMirror - Confusion Matrix", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#b2bec3")
    ax.yaxis.label.set_color("#b2bec3")
    ax.xaxis.label.set_color("#b2bec3")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ensure_table()

    days = None
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            print(f"Usage: python ar.py [days]")
            sys.exit(1)

    result = evaluate_accuracy(days)
    if result and HAS_PLOT:
        y_true, y_pred = result
        plot_confusion_matrix(y_true, y_pred)