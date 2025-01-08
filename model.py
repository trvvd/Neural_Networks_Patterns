import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM_Model(nn.Module):
    """
    CNN+LSTM Model for EMG classification.
    - in_channels assumed = 1 if data is shaped (batch_size, num_features)
      We'll permute the input to [batch_size, 1, num_features] if necessary
    - LSTM for temporal dependencies
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_rate=0.5):
        super(CNN_LSTM_Model, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm = nn.LSTM(
            input_size=128,  # must match out_channels from conv2
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Expects x of shape [batch_size, num_features] or [batch_size, seq_len, num_channels].
        For simplicity, we'll assume x is [batch_size, num_features], then reshape to [batch_size, 1, num_features].
        """
        # Reshape if necessary: (B, F) -> (B, 1, F)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # [batch_size, 1, num_features] -> [batch_size, 64, num_features]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Now x is [batch_size, 128, reduced_length]
        # Permute to feed into LSTM: [batch_size, reduced_length, 128]
        x = x.permute(0, 2, 1)

        # LSTM
        out, (h_n, c_n) = self.lstm(x)

        # Take the last output of LSTM
        # out is [batch_size, reduced_length, hidden_size]
        # so out[:, -1, :] is [batch_size, hidden_size]
        out = out[:, -1, :]

        # Fully connected layers
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out

# from model import CNN_LSTM_Model
# model = CNN_LSTM_Model(input_size=8, hidden_size=128, output_size=8, num_layers=3, dropout_rate=0.4)
# print(model)