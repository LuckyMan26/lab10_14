import streamlit as st

import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.1):
        super(LSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Improved classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.output = nn.Softmax(dim=-1)

    def forward(self, x):
        #h0 = torch.zeros(x.size(0), self.num_layers, self.hidden_size).to(x.device)
        #c0 = torch.zeros(x.size(0), self.num_layers,  self.hidden_size).to(x.device)
        #print("h0:", h0.shape)
        lstm_out, (h_n, c_n) = self.lstm(x)
        #print("lstm_out:", lstm_out.shape)
        final_hidden_state = lstm_out.reshape(lstm_out.size(0), lstm_out.size(1))
        print("final_hidden_state:", final_hidden_state.shape)
        print("layer_norm:", self.hidden_size)
        #print("final_hidden_state:", final_hidden_state.shape)
        pooled = self.layer_norm(final_hidden_state)
        pooled = self.dropout(pooled)

        # Classifier head
        out = self.classifier(pooled)
        #print("out:", pooled.shape)
        return self.output(out)

import torch
vocab_size = 10000
embedding_dim = 384
hidden_dim = 64
output_dim = 3  # For binary classification
sequence_length = 100

model = LSTMClassifier(vocab_size=vocab_size, embedding_dim=embedding_dim,hidden_dim = 32, output_dim = output_dim, num_layers=1, dropout=0.3)
model.load_state_dict(torch.load("lstm.pth", map_location='cuda'))
model.eval()

model = model.to('cuda')
# Load the embedding model
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2").to('cuda')

# Label mapping (example)
label_map = {0: "negative", 1: "neutral", 2: "positive"}

def predict_sentiment(text):
    inputs = embedding_model.encode(text, convert_to_tensor=True).unsqueeze(0).to('cuda')
    with torch.no_grad():
        print("inputs:", inputs.shape)
        outputs = model(inputs)
        prediction = torch.argmax(outputs, dim=1).item()
    return label_map[prediction]


st.title("Sentiment Analysis")

# Text input field
user_input = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: {sentiment}")
    else:
        st.error("Please enter some text to analyze.")