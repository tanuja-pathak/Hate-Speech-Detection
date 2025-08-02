import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from preprocessor import Preprocessor
from Model import TweetLSTMClassifier
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pickle

class TweetDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def main():
    # Preprocess data
    preprocessor = Preprocessor()
    processed_data = preprocessor.process(r'Hate_speech_detection_using_NLP_and_LSTM\labeled_data.csv')
    
    # Prepare tensors
    X = np.array(processed_data['padded_ids'].tolist())
    y = processed_data['class'].values
    
    # Balance classes
    smote = SMOTE(sampling_strategy='minority')
    X, y = smote.fit_resample(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Create dataloaders
    train_dataset = TweetDataset(X_train, y_train)
    test_dataset = TweetDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # Save the tensors instead of the DataLoader
    with open('test_loader_data.pkl', 'wb') as f:
        pickle.dump({
            'X_test': X_test,
            'y_test': y_test
        }, f)

    
    # Load vocabulary
    with open('vocab.pkl', 'rb') as f:
        vocab, MAX_LEN = pickle.load(f)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TweetLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=50,
        hidden_dim=100,
        output_dim=3
    ).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Acc: {train_acc:.4f}')
    
    # Save model
    torch.save(model.state_dict(), 'model_weights.pth')

if __name__ == '__main__':
    main()