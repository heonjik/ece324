import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# Import dataset and dataloader directly from Image_to_tensor.py
from Image_to_tensor import dataset, my_collate_fn
from torch.utils.data import DataLoader, random_split

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


if __name__ == '__main__':
    # Settings
    cnn_output_size = 128
    hidden_size = 64
    num_classes = 10
    learning_rate = 0.001
    max_epochs = 50
    patience = 5  # early stopping patience
    num_batches_per_epoch = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split dataset into train/val
    val_split = 0.2
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=my_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=my_collate_fn)

    # Initialize model
    model = CNNRNNModel(cnn_output_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Tracking lists
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # ======== Train ========
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        train_batches = list(train_loader)
        random.shuffle(train_batches)
        train_batches = train_batches[:num_batches_per_epoch]

        progress_bar = tqdm(train_batches, desc=f"Epoch {epoch+1}/{max_epochs}", unit="batch")

        for videos, labels in progress_bar:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_batches)
        train_accuracy = 100 * correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_accuracy)

        # ======== Validation ========
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_accuracy)

        print(f"Epoch {epoch+1} Summary -> Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # ======== Early Stopping Check ========
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ======== Plotting ========
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()
