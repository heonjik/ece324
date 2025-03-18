import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

class CNNEncoder(nn.Module):
    def __init__(self, output_size):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.feature_size = 64 * 7 * 7
        self.fc = nn.Linear(self.feature_size, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNNRNNModel(nn.Module):
    def __init__(self, cnn_output_size, hidden_size, num_classes, num_layers=1):
        super(CNNRNNModel, self).__init__()
        self.cnn = CNNEncoder(cnn_output_size)
        self.rnn = nn.LSTM(input_size=cnn_output_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        cnn_out = self.cnn(x)  # (B*T, cnn_output_size)
        cnn_out = cnn_out.view(B, T, -1)  # (B, T, cnn_output_size)
        rnn_out, _ = self.rnn(cnn_out)  # (B, T, hidden_size)
        out = self.classifier(rnn_out[:, -1, :])  # Use last timestep output
        return out

# Instantiate model and dummy input
model = CNNRNNModel(cnn_output_size=128, hidden_size=64, num_classes=10)
dummy_input = torch.randn(2, 5, 1, 28, 28)

# Forward pass
output = model(dummy_input)

# Generate graph and render it as PNG
make_dot(output, params=dict(model.named_parameters())).render("CNNRNNModel_Architecture", format="png")
