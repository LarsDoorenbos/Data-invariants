import data
import numpy as np
import eval 
import argparse
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bs = 64

@torch.no_grad()
def get_latent_vectors(data, model):
    loader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=False, num_workers=8)
    model.eval()

    latent_vectors = []
    for cnt, x in enumerate(loader):
        x = x[0].to(device)
        latent_vectors.append(model(x))

    return torch.cat(latent_vectors).cpu().numpy()

def get_knn_dists(trainFeatures, features):
    model = NearestNeighbors(n_neighbors=2)

    model.fit(trainFeatures)
    dists = np.mean(model.kneighbors(features)[0], axis=1)
        
    return dists

def main(in_class, task):
    if task == 'uniclass':
        train, testIn, testOut = data.get_cifar10(0, in_class)
    elif task == 'unisuper':
        train, testIn, testOut = data.get_cifar100(0, in_class)
    elif task == 'shift-lowres':
        train, testIn, testOut = data.get_lowres_shift_data(0, in_class)

    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
    model.to(device)
    model.fc = nn.Identity()

    trainFeatures = get_latent_vectors(train, model)
    inFeatures = get_latent_vectors(testIn, model)
    outFeatures = get_latent_vectors(testOut, model)

    inScores = get_knn_dists(trainFeatures, inFeatures)
    outScores = get_knn_dists(trainFeatures, outFeatures)

    auc = eval.compute_auc(inScores, outScores)

    print("AUC: {:.2f}".format(auc*100))

    return auc

if __name__ == "__main__":        
    parser = argparse.ArgumentParser()
    parser.add_argument('--numExps', type=int, default=1)
    parser.add_argument("--task", type=str, default="uniclass")
    args = parser.parse_args()

    aucs = []
    for i in range(args.numExps):
        aucs.append(main(i, args.task))

    aucs = np.array(aucs)
    print('Average AUC:', np.mean(aucs))    
    print(aucs)        