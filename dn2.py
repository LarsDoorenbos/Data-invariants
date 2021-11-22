import data
import numpy as np
import eval 
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

def main():
    train, testIn, testOut = data.get_cifar10(0)

    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)
    model.to(device)
    model.fc = nn.Identity()

    trainFeatures = get_latent_vectors(train, model)
    inFeatures = get_latent_vectors(testIn, model)
    outFeatures = get_latent_vectors(testOut, model)

    inScores = get_knn_dists(trainFeatures, inFeatures)
    outScores = get_knn_dists(trainFeatures, outFeatures)

    auc = eval.compute_auc(inScores, outScores)

    print("AUC: {:.2f}".format(auc*100))

if __name__ == "__main__":        
    main()        