import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="🎬 IMDB Sentiment Analyzer | RNN + LLM",
    page_icon="🎥",
    layout="centered"
)

# ----------------------------
# CONSTANTS
# ----------------------------
MAX_LEN = 200
LLM_CONFIDENCE_THRESHOLD = 0.75

# ----------------------------
# LOAD RNN MODEL
# ----------------------------
@st.cache_resource
def load_rnn():
    model = tf.keras.models.load_model("imdb_model_rnn.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# ----------------------------
# LOAD LIGHTWEIGHT LLM (SAFE)
# ----------------------------
@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=120
    )

rnn_model, tokenizer = load_rnn()
llm = load_llm()

# ----------------------------
# UI
# ----------------------------
st.title("🎬 IMDB Sentiment Analysis")
st.markdown("""
**Hybrid ML System**
- 🔹 RNN for fast sentiment prediction
- 🔹 LLM for validation, explanation & refinement
""")

review = st.text_area("✍️ Enter a movie review", height=150)

# ----------------------------
# ANALYSIS
# ----------------------------
if st.button("🔍 Analyze Sentiment"):

    if not review.strip():
        st.warning("⚠️ Please enter a review.")
    else:
        # ----- RNN Prediction -----
        sequence = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")

        prob = rnn_model.predict(padded)[0][0]
        sentiment = "Positive" if prob > 0.5 else "Negative"
        confidence = prob if prob > 0.5 else 1 - prob

        # ----- Display RNN Result -----
        st.subheader("📊 RNN Prediction")
        st.success(f"**{sentiment}**")
        st.write(f"Confidence: `{confidence*100:.2f}%`")

        # ----- LLM Validation (Only if needed) -----
        if confidence < LLM_CONFIDENCE_THRESHOLD:
            st.subheader("🧠 LLM Validation")

            prompt = f"""
Review:
{review}

Initial sentiment: {sentiment}

Verify the sentiment and explain briefly.
"""

            llm_output = llm(prompt)[0]["generated_text"]
            st.markdown(llm_output)

        else:
            st.info("✅ High confidence prediction — LLM validation skipped.")

        if sentiment == "Positive":
            st.balloons()
        else:
            st.warning("👎 Negative sentiment detected.")
