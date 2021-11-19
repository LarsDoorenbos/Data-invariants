import torchvision
import torch
import torchvision.transforms as transforms

EfNetCrops = [224, 240, 260, 300, 380, 456, 528, 600]
EfNetShapes = [int(EfNetCrops[i] / EfNetCrops[0] * 256) for i in range(len(EfNetCrops))]

def get_data(efnet):
    transform = transforms.Compose([
        transforms.Resize(EfNetCrops[efnet]),
        transforms.CenterCrop(EfNetCrops[efnet]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    indList = [0]

    trainSet = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)

    train = torch.tensor([1 if trainSet.targets[i] in indList  else 0 for i in range(len(trainSet))])
    trainSet = torch.utils.data.Subset(trainSet, train.nonzero())

    testSetIn = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    test = torch.tensor([1 if testSetIn.targets[i] in indList  else 0 for i in range(len(testSetIn))])

    testSetIn = torch.utils.data.Subset(testSetIn, test.nonzero())

    testSetOut = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    test = torch.tensor([1 if testSetOut.targets[i] not in indList else 0 for i in range(len(testSetOut))])

    testSetOut = torch.utils.data.Subset(testSetOut, test.nonzero())
    testSetOut, _ = torch.utils.data.random_split(testSetOut, [1000, len(testSetOut)-1000], generator=torch.Generator().manual_seed(1))

    return trainSet, testSetIn, testSetOut