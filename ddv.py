import data
import numpy as np
import eval 
import torch
import torch.nn as nn
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bs = 32
LOG_2PI = np.log(2 * np.pi)
bandwidth = -2
dims = 16


def lossFunction(query, points, bandwidth):
    log_num_points = np.log(points.shape[0])
    log_bandwidth = np.log(10**bandwidth)
    ndims = int(points.shape[1])

    sqrdists = torch.sum((query[:, None] - points[None, :])**2, dim=2)
    logkde = torch.logsumexp(-sqrdists / (2 * (10**bandwidth)**2), dim=1)
    logkde = logkde - log_num_points - ndims * (log_bandwidth + LOG_2PI * 0.5)

    return -logkde


@torch.no_grad()
def get_latent_vectors(dataloader, model):
    model.eval()

    latent_vectors = []
    for cnt, x in enumerate(dataloader):
        x = x[0].to(device)
        latent_vectors.append(model(x))

    return torch.cat(latent_vectors)


def train_DDV(train, model):
    trainLoader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True, num_workers=8)

    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=10**-4)

    model.to(device)
    for epoch in range(3):
        running_loss=0

        model.eval()
        frozenVectors = get_latent_vectors(trainLoader, model)

        model.train()
        for cnt, x in enumerate(trainLoader):  
            optimizer.zero_grad()
            outputs = model(x[0].to(device))

            loss = torch.sum(lossFunction(outputs, frozenVectors, bandwidth))
            running_loss += loss

            loss.backward()
            optimizer.step()

        print(epoch, running_loss)

    return model, frozenVectors


@torch.no_grad()
def eval_DDV(testIn, testOut, model, frozenVectors):    
    testInLoader = torch.utils.data.DataLoader(testIn, batch_size=bs, shuffle=False, num_workers=8)
    testOutLoader = torch.utils.data.DataLoader(testOut, batch_size=bs, shuffle=False, num_workers=8)

    testInScores = np.array([])
    for cnt, x in enumerate(testInLoader):  
        image = x[0].to(device)
        outputs = model(image)
        
        loss = lossFunction(outputs, frozenVectors, bandwidth)
        testInScores = np.concatenate((testInScores, loss.cpu()))

    testOutScores = np.array([])
    for cnt, x in enumerate(testOutLoader):        
        outputs = model(x[0].to(device))
        loss = lossFunction(outputs, frozenVectors, bandwidth)

        testOutScores = np.concatenate((testOutScores, loss.cpu()))

    auc = eval.compute_auc(np.array(testInScores), np.array(testOutScores))
    print("AUC: {:.2f}".format(auc*100))


def main(in_class, task):
    if task == 'uniclass':
        train, testIn, testOut = data.get_cifar10(0, in_class)
    elif task == 'unisuper':
        train, testIn, testOut = data.get_cifar100(0, in_class)
    elif task == 'shift-lowres':
        train, testIn, testOut = data.get_lowres_shift_data(0, in_class)

    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)
    model.to(device)
    model.fc = nn.Linear(2048, dims)

    model, frozenVectors = train_DDV(train, model)
    eval_DDV(testIn, testOut, model, frozenVectors)


if __name__ == "__main__":        
    parser = argparse.ArgumentParser()
    parser.add_argument('--numExps', type=int, default=1)
    parser.add_argument("--task", type=str, default="uniclass")
    args = parser.parse_args()

    auc = 0
    for i in range(args.numExps):
        auc += main(i, args.task)        

    print('Average AUC:', auc / args.numExps)    