import torchvision
import torch
import torchvision.transforms as transforms
import numpy as np

EfNetCrops = [224, 240, 260, 300, 380, 456, 528, 600]
EfNetShapes = [int(EfNetCrops[i] / EfNetCrops[0] * 256) for i in range(len(EfNetCrops))]

# From https://github.com/ryanchankh/cifar100coarse
def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]

def get_cifar10(efnet, in_class = 0):
    transform = transforms.Compose([
        transforms.Resize(EfNetCrops[efnet]),
        transforms.CenterCrop(EfNetCrops[efnet]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    indList = [in_class]

    trainSet = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)

    train = torch.tensor([1 if trainSet.targets[i] in indList  else 0 for i in range(len(trainSet))])
    trainSet = torch.utils.data.Subset(trainSet, train.nonzero())

    testSetIn = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    test = torch.tensor([1 if testSetIn.targets[i] in indList  else 0 for i in range(len(testSetIn))])

    testSetIn = torch.utils.data.Subset(testSetIn, test.nonzero())

    testSetOut = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    test = torch.tensor([1 if testSetOut.targets[i] not in indList else 0 for i in range(len(testSetOut))])

    testSetOut = torch.utils.data.Subset(testSetOut, test.nonzero())
    testSetOut, _ = torch.utils.data.random_split(testSetOut, [1000, len(testSetOut)-1000], generator=torch.Generator().manual_seed(42))

    return trainSet, testSetIn, testSetOut

def get_cifar100(efnet, in_class = 0):
    transform = transforms.Compose([
        transforms.Resize(EfNetCrops[efnet]),
        transforms.CenterCrop(EfNetCrops[efnet]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    indList = [in_class]

    trainSet = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
    trainSet.targets = sparse2coarse(trainSet.targets)

    train = torch.tensor([1 if trainSet.targets[i] in indList  else 0 for i in range(len(trainSet))])
    trainSet = torch.utils.data.Subset(trainSet, train.nonzero())

    testSetIn = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
    testSetIn.targets = sparse2coarse(testSetIn.targets)
    test = torch.tensor([1 if testSetIn.targets[i] in indList  else 0 for i in range(len(testSetIn))])

    testSetIn = torch.utils.data.Subset(testSetIn, test.nonzero())

    testSetOut = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
    testSetOut.targets = sparse2coarse(testSetOut.targets)

    test = torch.tensor([1 if testSetOut.targets[i] not in indList else 0 for i in range(len(testSetOut))])

    testSetOut = torch.utils.data.Subset(testSetOut, test.nonzero())
    testSetOut, _ = torch.utils.data.random_split(testSetOut, [500, len(testSetOut)-500], generator=torch.Generator().manual_seed(42))

    return trainSet, testSetIn, testSetOut    

def get_lowres_shift_data(efnet, in_class = 0):
    transform = transforms.Compose([
        transforms.Resize(EfNetCrops[efnet]),
        transforms.CenterCrop(EfNetCrops[efnet]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainSet = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)

    testSetIn = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

    if in_class == 0:
        testSetOut = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
    else:
        testSetOut = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        testSetOut, _ = torch.utils.data.random_split(testSetOut, [10000, len(testSetOut)-10000], generator=torch.Generator().manual_seed(42))

    return trainSet, testSetIn, testSetOut