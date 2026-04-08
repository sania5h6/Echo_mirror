"""
EchoMirror - Emotion Visualization Module (v2)
Charts:
  1. Emotion frequency bar chart (colored by emotion)
  2. Daily emotion timeline (stacked area)
  3. Weekly emotion heatmap
  4. Confidence distribution per emotion
  5. Sentiment polarity trend (from conversations)
"""

import sqlite3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import sys

# Match EchoMirror's emotion colors (RGB 0-1)
EMOTION_COLORS = {
    "happy":    "#00FF80",
    "sad":      "#FF6464",
    "angry":    "#FF0000",
    "fear":     "#C800C8",
    "surprise": "#00C8FF",
    "disgust":  "#008000",
    "neutral":  "#C8C8C8",
}

ALL_EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]


def _connect():
    return sqlite3.connect("emotions.db")


def _fetch_emotions(days=30):
    """Fetch emotion sessions for the last N days."""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        SELECT timestamp, date, emotion, confidence, brightness, is_low_light
        FROM emotion_sessions WHERE date >= ?
        ORDER BY timestamp ASC
    """, (since,))
    rows = c.fetchall()
    conn.close()
    return rows


def _fetch_conversations(days=30):
    """Fetch conversation polarity data for the last N days."""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        SELECT date, polarity FROM conversations
        WHERE date >= ? AND role = 'user'
        ORDER BY date ASC
    """, (since,))
    rows = c.fetchall()
    conn.close()
    return rows


# ─────────────────────────────────────────
# CHART 1: Emotion Frequency
# ─────────────────────────────────────────
def plot_emotion_frequency(rows):
    if not rows:
        print("[Visualize] No emotion data to chart.")
        return

    emotions = [r[2] for r in rows]
    counts = Counter(emotions)

    labels = sorted(counts.keys(), key=lambda e: ALL_EMOTIONS.index(e) if e in ALL_EMOTIONS else 99)
    values = [counts[l] for l in labels]
    colors = [EMOTION_COLORS.get(l, "#888888") for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    bars = ax.bar(labels, values, color=colors, edgecolor="#2d3436", linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")

    ax.set_title("Emotion Frequency", color="white", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Emotion", color="#b2bec3", fontsize=12)
    ax.set_ylabel("Count", color="#b2bec3", fontsize=12)
    ax.tick_params(colors="#b2bec3")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#636e72")
    ax.spines["left"].set_color("#636e72")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────
# CHART 2: Daily Emotion Timeline
# ─────────────────────────────────────────
def plot_daily_timeline(rows):
    if not rows:
        return

    # Count emotions per day
    daily = defaultdict(lambda: Counter())
    for r in rows:
        daily[r[1]][r[2]] += 1

    dates = sorted(daily.keys())
    if len(dates) < 2:
        print("[Visualize] Need at least 2 days for timeline.")
        return

    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    # Stacked area chart
    bottom = np.zeros(len(dates))
    for emo in ALL_EMOTIONS:
        values = [daily[d].get(emo, 0) for d in dates]
        color = EMOTION_COLORS.get(emo, "#888888")
        ax.fill_between(date_objs, bottom, bottom + values,
                        alpha=0.7, color=color, label=emo)
        bottom += np.array(values)

    ax.set_title("Daily Emotion Timeline", color="white", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Detections", color="#b2bec3", fontsize=12)
    ax.tick_params(colors="#b2bec3")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.3,
              facecolor="#1a1a2e", edgecolor="#636e72", labelcolor="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#636e72")
    ax.spines["left"].set_color("#636e72")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────
# CHART 3: Confidence Distribution
# ─────────────────────────────────────────
def plot_confidence_distribution(rows):
    if not rows:
        return

    # Group confidences by emotion
    conf_by_emo = defaultdict(list)
    for r in rows:
        if r[3] is not None:
            conf_by_emo[r[2]].append(r[3])

    if not conf_by_emo:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    positions = []
    labels = []
    for i, emo in enumerate(ALL_EMOTIONS):
        if emo in conf_by_emo and conf_by_emo[emo]:
            positions.append(i)
            labels.append(emo)

    if not positions:
        return

    data = [conf_by_emo[l] for l in labels]
    colors = [EMOTION_COLORS.get(l, "#888888") for l in labels]

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=True, flierprops=dict(marker=".", markersize=3, color="#636e72"))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("white")

    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("white")
            line.set_linewidth(1)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title("Confidence Distribution by Emotion", color="white", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Confidence (%)", color="#b2bec3", fontsize=12)
    ax.tick_params(colors="#b2bec3")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#636e72")
    ax.spines["left"].set_color("#636e72")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────
# CHART 4: Sentiment Polarity Trend
# ─────────────────────────────────────────
def plot_polarity_trend(conv_rows):
    if not conv_rows:
        print("[Visualize] No conversation data for polarity trend.")
        return

    # Average polarity per day
    daily_pol = defaultdict(list)
    for date, pol in conv_rows:
        if pol is not None:
            daily_pol[date].append(pol)

    if not daily_pol:
        return

    dates = sorted(daily_pol.keys())
    avg_pols = [sum(daily_pol[d]) / len(daily_pol[d]) for d in dates]
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    # Color gradient: negative=red, neutral=gray, positive=green
    colors = []
    for p in avg_pols:
        if p > 0.1:
            colors.append("#55efc4")
        elif p < -0.1:
            colors.append("#ff7675")
        else:
            colors.append("#b2bec3")

    ax.bar(date_objs, avg_pols, color=colors, width=0.8, edgecolor="#2d3436")
    ax.axhline(y=0, color="#636e72", linewidth=0.8, linestyle="--")

    ax.set_title("Daily Sentiment Polarity", color="white", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Average Polarity", color="#b2bec3", fontsize=12)
    ax.tick_params(colors="#b2bec3")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#636e72")
    ax.spines["left"].set_color("#636e72")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def show_all_charts(days=30):
    """Generate and display all EchoMirror visualization charts."""
    print(f"[EchoMirror Visualize] Generating charts for last {days} days...\n")

    rows = _fetch_emotions(days)
    conv_rows = _fetch_conversations(days)

    if not rows and not conv_rows:
        print("[Visualize] No data found. Use EchoMirror first to generate data.")
        return

    print(f"  Emotion detections: {len(rows)}")
    print(f"  Conversation logs:  {len(conv_rows)}")

    charts = []
    if rows:
        f1 = plot_emotion_frequency(rows)
        if f1: charts.append(("Emotion Frequency", f1))

        f2 = plot_daily_timeline(rows)
        if f2: charts.append(("Daily Timeline", f2))

        f3 = plot_confidence_distribution(rows)
        if f3: charts.append(("Confidence Distribution", f3))

    if conv_rows:
        f4 = plot_polarity_trend(conv_rows)
        if f4: charts.append(("Sentiment Trend", f4))

    if charts:
        print(f"\n  Generated {len(charts)} chart(s). Close windows to exit.")
        plt.show()
    else:
        print("[Visualize] Not enough data for any charts.")


if __name__ == "__main__":
    days = 30
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            print(f"Usage: python visualize.py [days]  (default: 30)")
            sys.exit(1)
    show_all_charts(days)
