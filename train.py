import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_PATH = "/Users/parvarora/Downloads/Customer Complain Classifier/data/complaints.csv"   # <-- put your dataset here
TEXT_COL = "Consumer complaint narrative"
LABEL_COL = "Product"

def train():
    df = pd.read_csv(DATA_PATH)

    # Drop rows where the complaint narrative is missing
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])

    X, y = df[TEXT_COL], df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Report:\n", classification_report(y_test, preds))

    # Save
    joblib.dump(pipe, "artifacts/model.joblib")
    print("✅ Model saved to artifacts/model.joblib")

if __name__ == "__main__":
    train()
