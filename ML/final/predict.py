import sys
import torch
import joblib
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum


threshold = 0.0
with open("./final/threshold.txt", "r") as threshold_file:
    while True:
        line = threshold_file.readline()
        if not line:
            break
        threshold = float(line.strip())


class transaction_type(Enum):
    CASH_IN = 0
    CASH_OUT = 1
    DEBIT = 2
    PAYMENT = 3
    TRANSFER = 4


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


# 定義 SWAG 類
class SWAG(nn.Module):
    def __init__(self, base_model, no_cov_mat=True, max_num_models=5):
        super(SWAG, self).__init__()
        self.base_model = base_model
        self.no_cov_mat = no_cov_mat
        self.max_num_models = max_num_models
        self.registered = False
        self.models_collected = 0

        # 用於存儲模型參數的均值和平方均值
        self.mean = []
        self.sq_mean = []

    def register(self):
        # 註冊基礎模型的初始參數
        for param in self.base_model.parameters():
            self.mean.append(param.data.clone().detach())
            self.sq_mean.append(param.data.clone().detach() ** 2)
        self.registered = True

    def collect_model(self, model):
        if not self.registered:
            self.register()

        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                # 更新參數的均值
                self.mean[i].mul_(self.models_collected / (self.models_collected + 1.0))
                self.mean[i].add_(param.data / (self.models_collected + 1.0))

                # 更新參數的平方均值
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
                # 計算參數的標準差
                std_dev = torch.sqrt(torch.clamp(sq_mean - mean**2, min=1e-6))
                # 從參數分佈中採樣
                sampled_param = mean + scale * torch.randn_like(mean) * std_dev
                param.data.copy_(sampled_param)

    def forward(self, x):
        return self.base_model(x)


# Check number of arguments
if len(sys.argv) != 12:
    print("wrong number of arguments")
    sys.exit(1)

scaler = joblib.load("final/scaler.pkl")
label_encoder = joblib.load("final/label_encoder.pkl")

simple_model = Trade_Model()
simple_model.load_state_dict(torch.load("final/simple_model.pt"))
simple_model.eval()

swag_base_model = Trade_Model()
swag_model = SWAG(swag_base_model)
swag_model.load_state_dict(torch.load("final/swag_model.pt"))
swag_model.eval()


def predict_type(input_row, true_type):
    input_df = pd.DataFrame(
        [input_row],
        columns=[
            "step",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "orig_diff",
            "dest_diff",
            "surge",
            "freq_dest",
        ],
    )

    input_df = input_df[
        [
            "step",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "orig_diff",
            "dest_diff",
            "surge",
            "freq_dest",
        ]
    ]

    input_data_scaled = scaler.transform(input_df)
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

    with torch.no_grad():
        simple_model_output = simple_model(input_tensor)
        swag_model_output = swag_model(input_tensor)

    simple_max_prob = torch.softmax(simple_model_output, dim=1).max().item()
    swag_max_prob = torch.softmax(swag_model_output, dim=1).max().item()

    simple_predicted_type = simple_model_output.argmax(dim=1).item()
    swag_predicted_type = swag_model_output.argmax(dim=1).item()

    # 使用LabelEncoder將數值標籤轉換回原始字符串標籤
    simple_predicted_label = label_encoder.inverse_transform([simple_predicted_type])[0]
    swag_predicted_label = label_encoder.inverse_transform([swag_predicted_type])[0]

    true_type_encoded = label_encoder.transform([true_type])[0]

    # 判斷是否為OOD
    is_ood = (
        (simple_max_prob < threshold)
        or (swag_max_prob < threshold)
        or (simple_predicted_type != true_type_encoded)
        or (swag_predicted_type != true_type_encoded)
    )

    return simple_predicted_label, swag_predicted_label, is_ood


step = int(sys.argv[1])
amount = float(sys.argv[2])
oldbalanceOrg = float(sys.argv[3])
newbalanceOrig = float(sys.argv[4])
oldbalanceDest = float(sys.argv[5])
newbalanceDest = float(sys.argv[6])
orig_diff = int(sys.argv[7])
dest_diff = int(sys.argv[8])
surge = int(sys.argv[9])
freq_dest = int(sys.argv[10])

# Test data
test_row = {
    "step": step,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,
    "orig_diff": orig_diff,
    "dest_diff": dest_diff,
    "surge": surge,
    "freq_dest": freq_dest,
}
true_type = transaction_type(int(sys.argv[11])).name

simple_label, swag_label, is_ood = predict_type(test_row, true_type)

print(f"The Threshold Used In This Prediction: {threshold}")
print(f"True Model Predicted Type: {true_type}")
print(f"Simple Model Predicted Type: {simple_label}")
print(f"SWAG Model Predicted Type: {swag_label}")
print(f'This data is {"OOD" if is_ood else "in-distribution"} according to the  model.')

with open("./final/result.txt", "w") as file:
    file.write(str(int(is_ood)) + "\n")
sys.exit(0)
