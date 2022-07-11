
from sklearn.metrics import roc_auc_score
import numpy as np


def compute_auc(scoresIn, scoresOut):
    groundTruthIn = np.array([1 for i in range(len(scoresIn))])
    groundTruthOut = np.array([-1 for i in range(len(scoresOut))])

    groundTruth = np.append(groundTruthIn, groundTruthOut)

    scores = np.append(scoresIn, scoresOut)

    auroc = roc_auc_score(groundTruth, -1 * scores)
    
    return auroc