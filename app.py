from pathlib import Path
import pickle

import streamlit as st


APP_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    with (APP_DIR / "model.pkl").open("rb") as model_file:
        model = pickle.load(model_file)
    with (APP_DIR / "scalar.pkl").open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


st.set_page_config(page_title="Sentiment Analysis", page_icon=":movie_camera:")
st.title("Movie Review Sentiment Analysis")

try:
    model, vectorizer = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the sentiment model: {exc}")
    st.stop()

review = st.text_area("Enter your review")

if st.button("Predict", type="primary"):
    review = review.strip()
    if not review:
        st.warning("Please enter a movie review.")
    else:
        try:
            review_vector = vectorizer.transform([review])
            prediction = model.predict(review_vector)[0]
            if prediction == 0:
                st.error("Negative review")
            else:
                st.success("Positive review")
        except Exception as exc:
            st.error(f"Could not analyze the review: {exc}")
