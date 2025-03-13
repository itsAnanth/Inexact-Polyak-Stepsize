import torch
import torchvision
from torchvision.transforms import transforms

def getData(dataset: str, batch_size=64):
    
    if not hasattr(torchvision.datasets, dataset):
        raise ValueError("dataset does not exist")
    
    data = getattr(torchvision.datasets, dataset)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = data(root='./data', train=True, download=True, transform=transform)
    testset = data(root='./data', train=False, download=True, transform=transform)
    
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(trainset), shuffle=True)
    validationloader = torch.utils.data.DataLoader(valset, batch_size=len(valset), shuffle=False)
    
    return trainloader, trainset, testloader, testset, validationloader, valset