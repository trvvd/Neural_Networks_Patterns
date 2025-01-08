import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from dataset import EMGDataset
from torch.utils.data import DataLoader
from model import CNN_LSTM_Model


def evaluate_model(model_path, test_data, input_size=8, hidden_size=128, output_size=8,
                   num_layers=3, dropout_rate=0.4, batch_size=64, device=None):
    """
    Load the best model from disk, evaluate on the test set, and print metrics.
    Args:
        model_path (str): path to the saved state dict
        test_data (tuple): (X_test, y_test)
        ...
    Returns:
        metrics (dict): dictionary containing accuracy, f1, precision, recall
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test, y_test = test_data
    test_dataset = EMGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model & load weights
    model = CNN_LSTM_Model(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "recall": recall,
        "precision": precision
    }

    print("Test Accuracy: {:.4f}".format(accuracy))
    print("Test F1-Score: {:.4f}".format(f1))
    print("Test Recall: {:.4f}".format(recall))
    print("Test Precision: {:.4f}".format(precision))

    return metrics