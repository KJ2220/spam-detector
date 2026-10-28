import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

@st.cache_resource
def train():
    df = pd.read_csv(URL, sep="\t", header=None, names=["label", "message"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    model = MultinomialNB()
    model.fit(vectorizer.fit_transform(X_train), y_train)
    return vectorizer, model

vectorizer, model = train()

st.title("📩 Spam Detector")
st.write("Type a message and see if it looks like spam.")

text = st.text_area("Message")

if st.button("Check"):
    if text.strip():
        pred = model.predict(vectorizer.transform([text]))[0]
        if pred == 1:
            st.error("SPAM")
        else:
            st.success("NOT SPAM")
    else:
        st.warning("Please type a message first.")
