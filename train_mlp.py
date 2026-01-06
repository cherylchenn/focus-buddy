import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models.headpose_mlp import HeadPoseMLP, LandmarkDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
MODEL_PATH = "models/mlp.pth"

def load_data(npz_file="data/biwi_landmarks.npz"):
    data = np.load(npz_file)
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        data["X"], data["y"], test_size=0.2, random_state=42, stratify=data["y"]
    )

    train_ds = LandmarkDataset(Xtrain, ytrain)
    test_ds = LandmarkDataset(Xtest, ytest)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    return train_loader, test_loader, Xtrain.shape[1], len(np.unique(data["y"]))

def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    
    return total_loss / len(loader.dataset)

def validate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    return 100 * correct / total

def train_model(train_loader, test_loader, input_dim, num_classes):
    model = HeadPoseMLP(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        loss = train(model, train_loader, criterion, optimizer)
        acc = validate(model, test_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f} - Val Acc: {acc:.2f}%")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Training complete! Model saved to {MODEL_PATH}")
    return model

if __name__ == "__main__":
    train_loader, test_loader, input_dim, num_classes = load_data()
    print(f"Training on {len(train_loader.dataset)} samples, validating on {len(test_loader.dataset)} samples")
    model = train_model(train_loader, test_loader, input_dim, num_classes)