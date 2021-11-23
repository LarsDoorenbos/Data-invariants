# Data-invariants

Run uniclass (10 exps) using an EfficientNet-b4 with
```
python3 mahaad.py --numExps 10 --task uniclass --efnet 4 --architecture en
```
Or unisuper (20 exps) using a ResNet-152 with
```
python3 mahaad.py --numExps 20 --task unisuper --architecture rn152 --bs 64
```
Or shift-lowres (2 exps) using a ResNet-101 with
```
python3 mahaad.py --numExps 2 --task shift-lowres --architecture rn101
```

### Data
Other experiments require external datasets, which can be downloaded from: <br>
MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad <br>
NIH: https://github.com/rsummers11/CADLab/tree/master/CXR-Binary-Classifier <br>
DRD: https://www.kaggle.com/c/diabetic-retinopathy-detection <br>
DomainNet: http://ai.bu.edu/M3SDA/ <br>

### Code for other methods
Results for the other methods were obtained using their official implementations, available at: <br>
MKD: https://github.com/rohban-lab/Knowledge_Distillation_AD  <br>
MSCL: https://github.com/talreiss/Mean-Shifted-Anomaly-Detection <br>
MHRot: https://github.com/hendrycks/ss-ood <br>
SSD: https://github.com/inspire-group/SSD <br>
IC/HierAD: https://github.com/boschresearch/hierarchical_anomaly_detection <br>
Glow: https://github.com/y0ast/Glow-PyTorch <br>

### Figures
Code to recreate figures 6 & 7, as far as possible for standalone code, is provided in the figures folder.
