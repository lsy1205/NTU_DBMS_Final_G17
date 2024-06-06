import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import sys
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Check number of arguments
if len(sys.argv) != 4:
    print("wrong number of arguments")
    sys.exit(1)

configuration = {
    "epochs": int(sys.argv[1]),
    "lr": float(sys.argv[2]),
    "momentum": float(sys.argv[3]),
}

with open("./final/model_result.txt", "w") as start_file:
    start_file.write("Training model, with ")
    start_file.write("Epochs: " + str(configuration["epochs"]) + ", ")
    start_file.write("Learning Rate: " + str(configuration["lr"]) + ", ")
    start_file.write("Momentum: " + str(configuration["momentum"]) + "\n")

# Load data
paysim = pd.read_excel("./final/data.xlsx")

# Encode labels
label_encoder = LabelEncoder()
paysim["type"] = label_encoder.fit_transform(paysim["type"])

# Separate fraud and non-fraud data
paysim_fraud_0 = paysim[paysim["isFraud"] == 0]
paysim_fraud_1 = paysim[paysim["isFraud"] == 1]

# Prepare input and output data
output = paysim_fraud_0["type"]
input_data = paysim_fraud_0.drop(
    columns=["type", "nameOrig", "nameDest", "isFraud", "isFlaggedFraud", "Unnamed: 0"]
)

# Split data into training and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(
    input_data, output, test_size=0.8, random_state=42
)
X_test1, X_temp, y_test1, y_temp = train_test_split(
    X_temp, y_temp, test_size=0.6667, random_state=42
)
X_test2, X_test3, y_test2, y_test3 = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test1_scaled = scaler.transform(X_test1)
X_test2_scaled = scaler.transform(X_test2)
X_test3_scaled = scaler.transform(X_test3)

# Convert data to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test1_tensor = torch.tensor(X_test1_scaled, dtype=torch.float32)
X_test2_tensor = torch.tensor(X_test2_scaled, dtype=torch.float32)
X_test3_tensor = torch.tensor(X_test3_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test1_tensor = torch.tensor(y_test1.values, dtype=torch.long)
y_test2_tensor = torch.tensor(y_test2.values, dtype=torch.long)
y_test3_tensor = torch.tensor(y_test3.values, dtype=torch.long)

# Create datasets and dataloaders
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test1_dataset = TensorDataset(X_test1_tensor, y_test1_tensor)
test2_dataset = TensorDataset(X_test2_tensor, y_test2_tensor)
test3_dataset = TensorDataset(X_test3_tensor, y_test3_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test1_loader = DataLoader(test1_dataset, batch_size=batch_size)
test2_loader = DataLoader(test2_dataset, batch_size=batch_size)
test3_loader = DataLoader(test3_dataset, batch_size=batch_size)


# Define SWAG model class
class SWAG(nn.Module):
    def __init__(self, base_model, no_cov_mat=True, max_num_models=5):
        super(SWAG, self).__init__()
        self.base_model = base_model
        self.no_cov_mat = no_cov_mat
        self.max_num_models = max_num_models
        self.registered = False
        self.models_collected = 0
        self.mean = []
        self.sq_mean = []

    def register(self):
        for param in self.base_model.parameters():
            self.mean.append(param.data.clone().detach())
            self.sq_mean.append(param.data.clone().detach() ** 2)
        self.registered = True

    def collect_model(self, model):
        if not self.registered:
            self.register()
        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                self.mean[i].mul_(self.models_collected / (self.models_collected + 1.0))
                self.mean[i].add_(param.data / (self.models_collected + 1.0))
                self.sq_mean[i].mul_(
                    self.models_collected / (self.models_collected + 1.0)
                )
                self.sq_mean[i].add_(param.data**2 / (self.models_collected + 1.0))
        self.models_collected += 1

    def sample(self, scale=0.3):
        if self.models_collected == 0:
            raise ValueError("No models collected")
        with torch.no_grad():
            for i, param in enumerate(self.base_model.parameters()):
                mean = self.mean[i]
                sq_mean = self.sq_mean[i]
                std_dev = torch.sqrt(torch.clamp(sq_mean - mean**2, min=1e-6))
                sampled_param = mean + scale * torch.randn_like(mean) * std_dev
                param.data.copy_(sampled_param)

    def forward(self, x):
        return self.base_model(x)


# Define training and testing functions
def train(loader, model, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} ({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def test(loader, model):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    test_loss /= total
    accuracy = 100.0 * correct / total
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)"
    )
    return accuracy


# Define Trade_Model class
class Trade_Model(nn.Module):
    def __init__(self):
        super(Trade_Model, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Initialize models and optimizers
model = Trade_Model()
optimizer = torch.optim.SGD(
    model.parameters(), lr=configuration["lr"], momentum=configuration["momentum"]
)
swag_model = SWAG(model, no_cov_mat=True, max_num_models=10)

# Train models
epochs = configuration["epochs"]
for epoch in range(epochs):
    train(train_loader, model, optimizer, epoch)
    swag_model.collect_model(model)

# Sample from SWAG model
swag_model.sample(0.1)

# Train simple model
simple_model = Trade_Model()
optimizer_simple = torch.optim.SGD(
    simple_model.parameters(),
    lr=configuration["lr"],
    momentum=configuration["momentum"],
)
for epoch in range(epochs):
    train(train_loader, simple_model, optimizer_simple, epoch)


# Configurations
print("The Number of Epochs:", configuration["epochs"])
print("The Learning Rate:", configuration["lr"])
print("The Momentum:", configuration["momentum"])
# Test models
print("Simple Model Result")
test(test1_loader, simple_model)
test(test2_loader, simple_model)
test(test3_loader, simple_model)
print("Swag Model Result")
test(test1_loader, swag_model)
test(test2_loader, swag_model)
test(test3_loader, swag_model)

# Save the models
torch.save(swag_model.state_dict(), "./final/swag_model.pt")
torch.save(simple_model.state_dict(), "./final/simple_model.pt")
## 可以調整的 epoch, learning rate  momentum (151-165行左右)
# with open("./final/model_result.txt", "w") as finish_file:
#     finish_file.write("Finish creating model, with")
#     finish_file.write("Epochs: " + str(configuration["epochs"]) + ", ")
#     finish_file.write("Learning Rate: " + str(configuration["lr"]) + ", ")
#     finish_file.write("Momentum: " + str(configuration["momentum"]) + "\n")
