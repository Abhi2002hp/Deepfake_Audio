import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from model import CNN_LSTM_Model

train_data = AudioDataset("dataset/training")
val_data = AudioDataset("dataset/validation")

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)

model = CNN_LSTM_Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(15):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/cnn_lstm_model.pth")