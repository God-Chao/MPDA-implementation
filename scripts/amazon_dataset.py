from torch.utils.data import Dataset, TensorDataset
import torch

class AmazonDataset(Dataset):
    def __init__(self, raw_data):
        self.user_id = torch.tensor(raw_data['user_id'].values)
        self.item_id = torch.tensor(raw_data['item_id'].values)
        self.rating = torch.tensor(raw_data['rating'].values)
        self.timestamp = torch.tensor(raw_data['timestamp'].values)
        self.label = (self.rating >= 4).to(dtype=torch.float32)
        self.dataset = TensorDataset(self.user_id, self.item_id, self.label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

