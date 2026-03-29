import sqlite3
from sklearn.metrics import classification_report, accuracy_score

def create_table():
    conn = sqlite3.connect("emotions.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS evaluation (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predicted TEXT,
        actual TEXT
    )
    """)

    conn.commit()
    conn.close()

def evaluate_accuracy():
    conn = sqlite3.connect("emotions.db")
    c = conn.cursor()

    c.execute("SELECT predicted, actual FROM evaluation")
    data = c.fetchall()

    conn.close()

    if not data:
        print("No evaluation data yet.")
        return

    y_pred = [d[0] for d in data]
    y_true = [d[1] for d in data]

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    create_table()
    evaluate_accuracy()