# Imports
import torch.nn.functional as F  # All functions that don't have any parameters
import pandas as pd
import torch
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from pandas import io
from tqdm import tqdm

# from skimage import io
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions

# Create Fully Connected Network
class HearthDiseaseNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HearthDiseaseNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, num_classes)
        # self.fc2 = nn.Linear(50, 40)
        # self.fc3 = nn.Linear(40, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = self.fc2(x)
        return x

class HearthDiseaseDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        x_data = self.annotations.iloc[index, 0:13]
        x_data = torch.tensor(x_data)
        y_label = torch.tensor(int(self.annotations.iloc[index, 13]))

        return (x_data.float(), y_label)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Hyperparameters
num_classes = 2
learning_rate = 1e-3
batch_size = 300
num_epochs = 600
input_size = 13

# Load Data
dataset = HearthDiseaseDataset(
    csv_file="heart.csv", root_dir="dataset", transform=transforms.ToTensor()
)

train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size

train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
model = HearthDiseaseNN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(len(train_set))
print(len(test_set))
# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)}")

# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )

    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)

