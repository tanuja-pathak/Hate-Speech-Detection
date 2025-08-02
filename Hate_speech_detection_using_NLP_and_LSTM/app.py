import torch
import spacy
import pickle
import gradio as gr
from Model import TweetLSTMClassifier

# Load preprocessing artifacts
with open('vocab.pkl', 'rb') as f:
    vocab, MAX_LEN = pickle.load(f)

nlp = spacy.load('en_core_web_sm')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = TweetLSTMClassifier(
    vocab_size=len(vocab),
    embedding_dim=50,
    hidden_dim=100,
    output_dim=3
).to(device)
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.eval()

# Preprocessing functions
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    
    # Pad sequence
    if len(token_ids) >= MAX_LEN:
        token_ids = token_ids[:MAX_LEN]
    else:
        token_ids += [vocab['<PAD>']] * (MAX_LEN - len(token_ids))
    
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

# Prediction function
def predict(text):
    input_tensor = preprocess_text(text)
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
    
    label_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
    return label_map[pred.item()]

# Gradio interface
gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Enter tweet..."),
    outputs="text",
    title="Hate Speech Detection",
    description="Classify tweets as: Hate Speech, Offensive Language, or Neither",
    examples=[
        ["I hate this community!"],
        ["You people are disgusting"],
        ["Why are you such an idiot?"]
    ]
).launch(share=True)