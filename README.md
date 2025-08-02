# Hate Speech Detection System

This project classifies tweets into three categories:
- Hate Speech (0)
- Offensive Language (1)
- Neither (2)

## Setup

1. Install requirements:

pip install -r requirements.txt

## Download spaCy model:

python -m spacy download en_core_web_sm

## Workflow

Preprocessing: preprocessor.py
Training: train.py
Evaluation: test.py
App: app.py

## Dataset
Place your labeled_data.csv in the project root. The dataset should contain:

tweet: Text content

class: Label (0, 1, or 2)

## Project Structure
preprocessor.py: Data cleaning and preprocessing

model.py: LSTM model architecture

train.py: Training pipeline

test.py: Model evaluation

app.py: Gradio web interface

vocab.pkl: Saved vocabulary

model_weights.pth: Trained model weights

## Results
Model achieves ~96% accuracy after 10 epochs and 80% in testing