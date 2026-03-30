import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from model import CNN_LSTM_Model
from sklearn.metrics import accuracy_score

test_data = AudioDataset("dataset/testing")
test_loader = DataLoader(test_data, batch_size=8)

model = CNN_LSTM_Model()
model.load_state_dict(torch.load("models/cnn_lstm_model.pth"))
model.eval()

preds, labels = [], []

with torch.no_grad():
    for x, y in test_loader:
        output = model(x)
        pred = output.argmax(1)

        preds.extend(pred.numpy())
        labels.extend(y.numpy())

print("Test Accuracy:", accuracy_score(labels, preds))