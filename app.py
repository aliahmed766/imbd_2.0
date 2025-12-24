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
# LOAD RNN MODEL
# ----------------------------
@st.cache_resource
def load_rnn():
    model = tf.keras.models.load_model("imdb_model_rnn.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# ----------------------------
# LOAD LLM
# ----------------------------
@st.cache_resource
def load_llm():
    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-3B-Instruct",
        device_map="auto",
        temperature=0.2,
        max_new_tokens=180
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
# INFERENCE
# ----------------------------
if st.button("🔍 Analyze Sentiment"):

    if not review.strip():
        st.warning("Please enter a review.")
    else:
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=200, padding="post")

        prob = rnn_model.predict(padded)[0][0]
        sentiment = "Positive" if prob > 0.5 else "Negative"
        confidence = prob if prob > 0.5 else 1 - prob

        prompt = f"""
You are a sentiment analysis expert.

Review:
\"\"\"{review}\"\"\"

Initial Prediction:
Sentiment: {sentiment}
Confidence: {confidence:.2f}

Tasks:
1. Verify sentiment correctness
2. Correct only if clearly wrong
3. Provide a short explanation (2–3 lines)

Answer format:
Sentiment:
Explanation:
"""

        llm_result = llm(prompt)[0]["generated_text"]

        st.subheader("📊 RNN Prediction")
        st.success(f"**{sentiment}**")
        st.write(f"Confidence: {confidence*100:.2f}%")

        st.subheader("🧠 LLM Refined Output")
        st.markdown(llm_result)

        if sentiment == "Positive":
            st.balloons()
