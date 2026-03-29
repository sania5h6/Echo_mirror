import sqlite3
import matplotlib.pyplot as plt
from collections import Counter

def show_emotion_chart():
    conn = sqlite3.connect("emotions.db")
    c = conn.cursor()

    c.execute("SELECT emotion FROM emotions")
    data = c.fetchall()
    conn.close()

    emotions = [d[0] for d in data]

    if len(emotions) == 0:
        print("No data to visualize yet.")
        return

    counts = Counter(emotions)

    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(8,5))
    plt.bar(labels, values)
    plt.title("EchoMirror Emotion Frequency")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
