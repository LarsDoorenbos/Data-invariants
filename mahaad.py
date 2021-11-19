from sklearn.covariance import LedoitWolf
import torch
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import data
import numpy as np
import eval 

efnet = 0
bs = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Adapted from https://github.com/byungjae89/MahalanobisAD-pytorch
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
            if (x_prev.shape[1] != x.shape[1] and idx != 0) or idx == (len(self._blocks) - 1):
                features.append(x_prev.mean(dim=(2,3)))
            x_prev = x

        x = self._swish(self._bn1(self._conv_head(x)))
        features.append(x.mean(dim=(2,3)))

        return features

@torch.no_grad()
def get_latent_vectors(data, model):
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

def get_maha_dists(train, points):
    scores = np.zeros(points[str(0)].shape[0])

    for layer in range(len(train)):       
        train[str(layer)] = np.array(train[str(layer)])
        mean = np.mean(train[str(layer)], axis=0)

        LW = LedoitWolf().fit(train[str(layer)])

        # Typically np linalg inv gives slightly better results than LW.precision_
        # covI = LW.precision_
        covI = np.linalg.inv(LW.covariance_)

        points[str(layer)] = (points[str(layer)] - mean)[:, None]
        dists = covI @ points[str(layer)].transpose(0, 2, 1)
        dists = points[str(layer)] @ dists
        dists = np.sqrt(dists[:, 0, 0])    

        scores += dists

    return scores   

def main():
    train, testIn, testOut = data.get_data(efnet)

    model = EfficientNet_features.from_pretrained('efficientnet-b' + str(efnet))
    model.to(device)

    trainFeatures = get_latent_vectors(train, model)
    inFeatures = get_latent_vectors(testIn, model)
    outFeatures = get_latent_vectors(testOut, model)
    
    inScores = get_maha_dists(trainFeatures, inFeatures)
    outScores = get_maha_dists(trainFeatures, outFeatures)

    auc = eval.compute_auc(inScores, outScores)

    print("AUC: {:.2f}".format(auc*100))

if __name__ == "__main__":        
    main()        