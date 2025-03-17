import torch
import torchvision
from torchvision.transforms import transforms

def getData(dataset: str, transform_train, transform_val, batch_size=64):
    
    if not hasattr(torchvision.datasets, dataset):
        raise ValueError("dataset does not exist")
    
    data = getattr(torchvision.datasets, dataset)
    
    # Create datasets without transforms initially
    full_trainset = data(root='../data', train=True, download=True, transform=None)
    testset = data(root='../data', train=False, download=True, transform=transform_val)
    
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    
    # Split without transforms
    train_subset, val_subset = torch.utils.data.random_split(full_trainset, [train_size, val_size])
    
    # Create dataset wrappers that apply transforms
    trainset = TransformDataset(train_subset, transform_train)
    valset = TransformDataset(val_subset, transform_val)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    validationloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    
    return trainloader, trainset, testloader, testset, validationloader, valset

# Helper class to apply transforms to a subset
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)