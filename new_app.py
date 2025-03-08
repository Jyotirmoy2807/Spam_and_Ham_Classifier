import os
import gdown
import streamlit as st
import nltk
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure necessary NLTK resources are available
# nltk.data.path.append("C:/Users/Joydip/nltk_data")
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

MODEL_FILES = {
    "lstm": "15u3TCsOy1sghN35t3biTkM7O6SqHduFC",  # Replace with actual file ID for LSTM model
    "tokenizer": "1-30109fVOS3SOvVtCE2km7MIsirG-bzu",  # Replace with tokenizer file ID
    "classical_ml": "1dAD0LovytQUDlmSxwD0y9de4CcgkLhr3",  # Bernoulli Naive Bayes model file ID
    "fcnn": "1-58vncyW_FbjI0OE1ixaHve1Pe2Oi_K_",  # FCNN model file ID
    "tfidf": "1-7VoSQi8P8xKDQfNxMLo88PIEVzbQ9sj"  # TF-IDF vectorizer file ID
}

def download_file(file_id, file_name):
    if not os.path.exists(file_name):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", file_name, quiet=False)

@st.cache_resource
# def Load_Model():
#     """Load models and required resources."""
#     lstm_model = load_model('C:/Users/Joydip/OneDrive/Desktop/College Assignmets_Material/INSTRU/spam_classifier_Blstm.h5')
    
#     with open('C:/Users/Joydip/OneDrive/Desktop/College Assignmets_Material/INSTRU/tokenizer.pickle', 'rb') as handle:
#         tokenizer = pickle.load(handle)
    
#     classical_ml = pickle.load(open('C:/Users/Joydip/OneDrive/Desktop/College Assignmets_Material/INSTRU/bernoulli_naive_bayes_model.sav', 'rb'))
#     fcnn = load_model('C:/Users/Joydip/OneDrive/Desktop/College Assignmets_Material/INSTRU/fcnn_combined_model_etc.h5')
    
#     with open('C:/Users/Joydip/OneDrive/Desktop/College Assignmets_Material/INSTRU/tfidf_vectorizer.pkl', 'rb') as handle:
#         tfidf = pickle.load(handle)
    
#     return lstm_model, classical_ml, fcnn, tokenizer, tfidf


def Load_Model():
    # Download models if not present
    download_file(MODEL_FILES["lstm"], "spam_classifier_Blstm.h5")
    download_file(MODEL_FILES["tokenizer"], "tokenizer.pickle")
    download_file(MODEL_FILES["classical_ml"], "bernoulli_naive_bayes_model.sav")
    download_file(MODEL_FILES["fcnn"], "fcnn_combined_model_etc.h5")
    download_file(MODEL_FILES["tfidf"], "tfidf_vectorizer.pkl")

    # Load models
    lstm_model = load_model("spam_classifier_Blstm.h5")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    classical_ml = pickle.load(open("bernoulli_naive_bayes_model.sav", "rb"))
    fcnn = load_model("fcnn_combined_model_etc.h5")
    with open("tfidf_vectorizer.pkl", "rb") as handle:
        tfidf = pickle.load(handle)

    return lstm_model, classical_ml, fcnn, tokenizer, tfidf


def preprocess_input(ip_text):
    """Preprocess input text for the model."""
    import string
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem.porter import PorterStemmer

    ps = PorterStemmer()

    text = ip_text.lower()
    text = word_tokenize(text)

    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]

    return " ".join(text)

def predict_sms(sms):
    """Predict whether an SMS is spam or ham."""
    lstm_model, classical_ml, fcnn, tokenizer, tfidf = Load_Model()

    # LSTM Model Processing
    sequence = tokenizer.texts_to_sequences([sms])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    lstm_prob = lstm_model.predict(padded_sequence)[0, 0]

    # Classical Model Processing
    transformed_sms = preprocess_input(sms)
    tfidf_features = tfidf.transform([transformed_sms]).toarray()
    classical_prob = classical_ml.predict_proba(tfidf_features)[:, 1][0]

    # FCNN Final Decision
    combined_features = np.hstack(([[lstm_prob]], [[classical_prob]]))
    final_prob = fcnn.predict(combined_features)[0, 0]

    prediction = "🚨 **Spam**" if final_prob > 0.5 else "✅ **Ham**"
    
    return {
        "LSTM Probability": lstm_prob,
        "Classical Probability": classical_prob,
        "Final Probability": final_prob,
        "Prediction": prediction
    }

# Streamlit UI
st.title("📩 SMS Spam Detector")
st.write("Enter a message below to check if it's spam or not.")

# User Input
user_input = st.text_area("✍️ **Enter your message:**", height=100)

if st.button("🔍 **Analyze Message**"):
    if user_input.strip():
        prediction = predict_sms(user_input)

        # Display Prediction with Color
        if "Spam" in prediction["Prediction"]:
            st.error(f"🚨 **Warning! This message is classified as Spam!**")
        else:
            st.success(f"✅ **This message is safe (Not Spam).**")

        # Expandable Section for Detailed Probabilities
        with st.expander("📊 **Detailed Probability Breakdown**"):
            st.write(f"📌 **LSTM Model Probability:** `{prediction['LSTM Probability']:.4f}`")
            st.write(f"📌 **Classical Model Probability:** `{prediction['Classical Probability']:.4f}`")
            st.write(f"📌 **Final Decision Probability:** `{prediction['Final Probability']:.4f}`")

    else:
        st.warning("⚠️ Please enter a valid message to analyze.")
