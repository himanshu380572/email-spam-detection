# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score

# # Load dataset
# df = pd.read_csv("spam mail.csv")

# # Features and labels
# emails = df["Masseges"]
# labels = df["Category"]

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     emails,
#     labels,
#     test_size=0.2,
#     random_state=42,
#     stratify=labels
# )

# # Create pipeline
# model = Pipeline([
#     ("tfidf", TfidfVectorizer(stop_words="english")),
#     ("clf", LogisticRegression(max_iter=1000))
# ])

# # Train
# model.fit(X_train, y_train)

# # Predict
# predictions = model.predict(X_test)

# # Results
# print("Accuracy:", accuracy_score(y_test, predictions))
# print("\nClassification Report:\n")
# print(classification_report(y_test, predictions))

# df = pd.read_csv("spam mail.csv")

# df["Prediction"] = model.predict(df["Masseges"])

# print(df.head())

import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("📧 Email Spam Detection")

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(df.head())

    text_col = st.selectbox(
        "Select Text Column",
        df.columns
    )

    label_col = st.selectbox(
        "Select Label Column",
        df.columns
    )

    if st.button("Train Model"):

        emails = df[text_col]
        labels = df[label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            emails,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        model = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000))
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        st.success(
            f"Model Trained Successfully! Accuracy = {accuracy:.2%}"
        )

        st.session_state["model"] = model

if "model" in st.session_state:

    st.subheader("Check Email")

    user_email = st.text_area(
        "Enter Email Text"
    )

    if st.button("Predict"):

        result = st.session_state["model"].predict(
            [user_email]
        )[0]

        if str(result).lower() == "spam":
            st.error("🚨 SPAM EMAIL")
        else:
            st.success("✅ HAM / NOT SPAM")