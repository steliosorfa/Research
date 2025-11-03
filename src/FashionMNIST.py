import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 1) Data (μόνο 1.000 δείγματα για πολύ γρήγορη δοκιμή)
tfm = transforms.ToTensor()
train_full = datasets.FashionMNIST(root='data', train=True, download=True, transform=tfm)
## train = Subset(train_full, range(1000))
train = train_full
test  = datasets.FashionMNIST(root='data', train=False, download=True, transform=tfm)

train_loader = DataLoader(train, batch_size=128, shuffle=True)
test_loader  = DataLoader(test,  batch_size=256, shuffle=False)

# 2) Μικρό MLP
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3) Train
for epoch in range(10):
    model.train()
    running = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        running += loss.item()*y.size(0)
    print(f"Epoch {epoch+1}: loss = {running/len(train):.4f}")

# 4) Test
model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
print(f"Test accuracy: {correct/total:.3f}")


# Δες μερικές προβλέψεις
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x, y = next(iter(test_loader))
with torch.no_grad():
    preds = model(x).argmax(1)

plt.figure(figsize=(8, 4))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(x[i].squeeze(), cmap='gray')
    plt.title(f'Pred: {classes[preds[i]]}\nTrue: {classes[y[i]]}')
    plt.axis('off')
plt.tight_layout()
plt.show()