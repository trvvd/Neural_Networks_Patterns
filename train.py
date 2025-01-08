import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from dataset import EMGDataset
from model import CNN_LSTM_Model


def train_model(
        train_data,
        val_data,
        input_size=8,
        hidden_size=128,
        output_size=8,
        num_layers=3,
        dropout_rate=0.4,
        batch_size=64,
        epochs=20,
        lr=0.0005,
        device=None
):
    """
    Train the CNN-LSTM model with the given dataset.
    Args:
        train_data (tuple): (X_train, y_train)
        val_data (tuple): (X_val, y_val)
        ...
    Returns:
        model: trained PyTorch model
        train_history: dictionary with losses, accuracies, etc.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = train_data
    X_val, y_val = val_data

    # Create Datasets and Loaders
    train_dataset = EMGDataset(X_train, y_train)
    val_dataset = EMGDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = CNN_LSTM_Model(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    ).to(device)

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # For logging
    train_history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": []
    }

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_valb, y_valb in val_loader:
                X_valb, y_valb = X_valb.to(device), y_valb.to(device)
                preds = model(X_valb)

                loss_valb = criterion(preds, y_valb)
                val_loss += loss_valb.item()

                _, predicted = torch.max(preds, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_valb.cpu().numpy())

        epoch_train_loss = running_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)

        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        train_history["train_loss"].append(epoch_train_loss)
        train_history["val_loss"].append(epoch_val_loss)
        train_history["val_accuracy"].append(val_accuracy)
        train_history["val_f1"].append(val_f1)

        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Epoch Time: {epoch_time:.2f}s")

        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")

    return model, train_history

# from train import train_model
# (X_train, y_train) = ...
# (X_val, y_val) = ...
# model, history = train_model((X_train, y_train), (X_val, y_val), epochs=10)