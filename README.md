# 🎬 IMDB Sentiment Analysis using RNN + Streamlit

A deep learning project that classifies **IMDB movie reviews** as **Positive 😊** or **Negative 😡** using a **Recurrent Neural Network (RNN)** built with TensorFlow/Keras.
This app provides an **interactive Streamlit interface** to test real-time sentiment predictions.

---

## 🚀 Features

* 🧠 Built using **Recurrent Neural Network (RNN)**
* 📈 Achieved **80%+ test accuracy** on the IMDB dataset
* 🔤 Custom **tokenizer** and text preprocessing pipeline
* 💾 Model saved as `.h5` and tokenizer as `.pkl`
* 🌐 Interactive **Streamlit web app** for real-time predictions
* 📊 Visualization of accuracy and loss trends
* 🔍 **Confusion matrix** visualization for evaluation

---

## 🧩 Project Structure

```
📦 imdb-sentiment-rnn-streamlit
 ┣ 📜 app.py                 # Streamlit app file
 ┣ 📜 imdb_model_rnn.h5      # Trained RNN model
 ┣ 📜 tokenizer.pkl          # Tokenizer file
 ┣ 📜 imdb_training.ipynb    # Model training notebook
 ┣ 📜 requirements.txt       # Dependencies
 ┗ 📜 README.md              # Project documentation
```

---

## ⚙️ Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/<your-username>/imdb-sentiment-rnn-streamlit.git
   cd imdb-sentiment-rnn-streamlit
   ```

2. **Create a virtual environment**

   ```bash
   conda create -n sentiment python=3.10
   conda activate sentiment
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Training Summary

| Metric              |         Value        |
| :------------------ | :------------------: |
| Training Accuracy   |          85%         |
| Validation Accuracy |          82%         |
| Test Accuracy       |       **81.4%**      |
| Model Type          | SimpleRNN (3 layers) |
| Optimizer           |         Adam         |
| Loss Function       |  Binary Crossentropy |

---

## 💡 Run the Streamlit App

1. Make sure your model and tokenizer files exist in the project directory:

   ```
   imdb_model_rnn.h5
   tokenizer.pkl
   ```

2. Run the app:

   ```bash
   streamlit run app.py
   ```

3. Visit:

   ```
   http://localhost:8501
   ```

   and test the model using your own movie reviews!

---

## 🧪 Example Predictions

| Review                                         | Prediction                  |
| :--------------------------------------------- | :-------------------------- |
| “The movie was fantastic and full of emotion.” | Positive                    |
| “I couldn’t finish it. The plot was terrible.” | Negative                    |
| “It had good acting but was a bit too long.”   | Neutral / Slightly Negative |

---

## 📊 Confusion Matrix

Visualizes the model’s performance on the test data:

```
True Positives / False Negatives / False Positives / True Negatives
```

---

## 🧰 Tech Stack

* **Python**
* **TensorFlow / Keras**
* **Streamlit**
* **Matplotlib / Seaborn**
* **Scikit-learn**
* **NumPy / Pandas**

---

## ✨ Future Improvements

* Add **LSTM/GRU** version for higher accuracy
* Expand to **multi-class emotion detection**
* Deploy on **Hugging Face Spaces / Streamlit Cloud**

---

## 🧑‍💻 Author

**Ali Ahmed**
💼 AI/ML Developer | 🎓 Deep Learning Enthusiast
📧 [[YourEmail@example.com](mailto:q707246@gmail.com)]
🌐 [Your LinkedIn or Portfolio link]

---

## 🪪 License

This project is open-source under the [MIT License](LICENSE).
