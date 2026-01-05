import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
MODEL_PATH = "headpose_mlp.pth"

class LandmarkDataset(Dataset): # dataset wrapper
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class HeadPoseMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

def load_data(npz_file="biwi_landmarks.npz"):
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
    
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
    
    return total_loss / len(loader.dataset)

def validate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_X)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    
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