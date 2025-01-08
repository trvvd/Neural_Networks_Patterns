import torch
from torch.utils.data import Dataset

class EMGDataset(Dataset):
    """
    Custom Dataset for EMG data.
    Expects features (X) and labels (y) as numpy arrays or
    something convertible to Torch tensors.
    """
    def __init__(self, features, labels):
        super().__init__()
        # Convert features and labels to PyTorch tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return a tuple: (feature_vector, label)
        return self.features[idx], self.labels[idx]


# from dataset import EMGDataset
# dataset = EMGDataset(X_train, y_train)
# print(len(dataset))
# print(dataset[0])