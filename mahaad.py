from numpy.core.fromnumeric import shape
from sklearn.covariance import LedoitWolf
import torch
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import data
import numpy as np
import eval 
import argparse
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Adapted from https://github.com/ORippler/gaussian-ad-mvtec
class EfficientNet_features(EfficientNet):
    def get_features(self, inputs):
        features = []

        x = self._swish(self._bn0(self._conv_stem(inputs)))
        features.append(x.mean(dim=(2,3)))

        x_prev = x
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if (x_prev.shape[1] != x.shape[1] and idx != 0):
                features.append(x_prev.mean(dim=(2,3)))
            if idx == (len(self._blocks) - 1):
                features.append(x.mean(dim=(2,3)))
            x_prev = x

        x = self._swish(self._bn1(self._conv_head(x)))
        features.append(x.mean(dim=(2,3)))

        return features

# Adapted from https://github.com/ORippler/gaussian-ad-mvtec
class ResNet_features(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.conv1 = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def get_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        result = []
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        for i in [0, 1, 2, 3, 4]:
            result.append(locals()["x" + str(i)].mean(dim=(2,3)))

        return result

@torch.no_grad()
def get_latent_vectors(data, model, bs):
    loader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=False, num_workers=8)
    model.eval()
    
    latent_vectors = {}
    
    for cnt, x in enumerate(loader):
        x = x[0].to(device)
        features = model.get_features(x)
        
        for i in range(len(features)):
            if cnt == 0:
                latent_vectors[str(i)] = []    

            latent_vectors[str(i)].append(features[i])

    for i in range(len(features)):
        latent_vectors[str(i)] = torch.cat(latent_vectors[str(i)]).cpu().numpy()

    return latent_vectors

def get_maha_dists(train, points, architecture):
    scores = np.zeros(points[str(0)].shape[0])
    
    for layer in range(len(train)):       
        train[str(layer)] = np.array(train[str(layer)])
        mean = np.mean(train[str(layer)], axis=0)

        LW = LedoitWolf().fit(train[str(layer)])

        # Typically np linalg inv gives slightly better results than LW.precision_, so EfNet results could probably be improved slightly further. ResNet results already w/ np.linalg.inv
        if architecture == 'en':
            covI = LW.precision_
        else:
            covI = np.linalg.inv(LW.covariance_)

        points[str(layer)] = (points[str(layer)] - mean)[:, None]
        dists = covI @ points[str(layer)].transpose(0, 2, 1)
        dists = points[str(layer)] @ dists
        dists = np.sqrt(dists[:, 0, 0])    

        scores += dists

    return scores   

def main(in_class, task, efnet, bs, architecture):
    if task == 'uniclass':
        train, testIn, testOut = data.get_cifar10(efnet, in_class)
    elif task == 'unisuper':
        train, testIn, testOut = data.get_cifar100(efnet, in_class)
    elif task == 'shift-lowres':
        train, testIn, testOut = data.get_lowres_shift_data(efnet, in_class)

    print(len(train), len(testIn), len(testOut))

    if architecture == 'en':
        model = EfficientNet_features.from_pretrained('efficientnet-b' + str(efnet))
    elif architecture == 'rn101':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
        model = ResNet_features(model)
    elif architecture == 'rn152':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)
        model = ResNet_features(model)

    model.to(device)

    trainFeatures = get_latent_vectors(train, model, bs)
    inFeatures = get_latent_vectors(testIn, model, bs)
    outFeatures = get_latent_vectors(testOut, model, bs)
    
    inScores = get_maha_dists(trainFeatures, inFeatures, architecture)
    outScores = get_maha_dists(trainFeatures, outFeatures, architecture)

    auc = eval.compute_auc(inScores, outScores)

    print("AUC: {:.2f}".format(auc*100))

    return auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--numExps', type=int, default=1)
    parser.add_argument('--efnet', type=int, default=0)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument("--task", type=str, default="uniclass")
    parser.add_argument("--architecture", type=str, default="en")
    args = parser.parse_args()

    auc = 0
    for i in range(args.numExps):
        auc += main(i, args.task, args.efnet, args.bs, args.architecture)        

    print('Average AUC:', auc / args.numExps)