import torch
from train import TweetDataset, DataLoader
from Model import TweetLSTMClassifier
import pickle
from sklearn.metrics import classification_report

def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return classification_report(all_labels, all_preds, target_names=['Hate', 'Offensive', 'Neither'])

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load vocabulary
    with open('vocab.pkl', 'rb') as f:
        vocab, MAX_LEN = pickle.load(f)
    
    # Load model
    model = TweetLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=50,
        hidden_dim=100,
        output_dim=3
    ).to(device)
    model.load_state_dict(torch.load('model_weights.pth'))
    
    # Load test data (should be saved during training)
    # This would typically come from your train.py
    # For simplicity, we assume test_loader is available
    # In practice, you'd save test data or regenerate it
    # Load saved test data
    with open('test_loader_data.pkl', 'rb') as f:
        data = pickle.load(f)
        X_test = data['X_test']
        y_test = data['y_test']

    # Recreate dataset and loader
    test_dataset = TweetDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluation
    report = evaluate(model, test_loader, device)
    print("Classification Report:")
    print(report)

if __name__ == '__main__':
    main()