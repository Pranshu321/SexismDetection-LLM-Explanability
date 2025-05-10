import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pickle
# from spike_neural import model as spike_model
import tensorflow as tf

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    "C:/Users/prans/Desktop/M.Tech Sem/AIDA Sem2/NN/Project/SexismLLM/saved_model")
tokenizer = AutoTokenizer.from_pretrained(
    "C:/Users/prans/Desktop/M.Tech Sem/AIDA Sem2/NN/Project/SexismLLM/saved_model")

# Load the Embeddings model
use_model_1 = tf.saved_model.load(
    "C:/Users/prans/Desktop/M.Tech Sem/AIDA Sem2/NN/Project/SexismLLM/saved_use_model")


def get_embeddings(text):
    return use_model_1(text).numpy()


def load_neural_model():
    with open('neural.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
        return loaded_model


def predict_using_neural_network(text):
    loaded_model = load_neural_model()
    text_embedding = get_embeddings([text])
    prediction = loaded_model.predict(text_embedding)
    return prediction[0][0] > 0.5


def preprocess(text):
    return tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")


# Function to predict a single text


# def predict_single(text):
#     # Convert text to embedding
#     text_embedding = get_embeddings([text])  # Ensure it's a NumPy array

#     # Convert to PyTorch tensor and move to device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     text_tensor = torch.tensor(text_embedding).float().to(device)

#     # Make prediction
#     spike_model.to(device)
#     spike_model.eval()

#     with torch.no_grad():
#         output = spike_model(text_tensor).squeeze(1)
#         # Convert logits to binary class
#         prediction = torch.sigmoid(output) > 0.5
#     print(prediction)
#     return prediction.item()  # Convert tensor to Python boolean


def predict(text):
    inputs = preprocess(text)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities.tolist()


def main():
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.radio(
        "Choose a model:", ("BERT Model", "Neural Network" , "Spike Neural Network"))

    html_temp = """
    <div style="background-color:green;padding:10px;margin-bottom:20px;">
    <h2 style="color:white;text-align:center;">Sexism Detection in Texts</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("Paste the text for prediction")
    # text = st.text_area("Text", "Type Here")

    text = st.text_area("Enter Tweet")

    if text:
        if st.button("Predict"):
            if model_choice in ["Neural Network", "Spike Neural Network"]:
                prediction = predict_using_neural_network(text)
                label = int(prediction)
            else:
                label, _ = predict(text)
            
            if label == 1:
                st.success("This is a Sexist Tweet")
            else:
                st.error("This is a Non Sexist Tweet")


if __name__ == '__main__':
    main()
