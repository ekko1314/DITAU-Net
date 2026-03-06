# DITAU-Net
# This repository is a supplement to our paper "DITAU-Net: Integrating Dual-Interactive Temporal Convolution with Adversarial and U-Shaped Autoencoders for Multivariate Time Series Anomaly Detection"

Preprocess all datasets using the command
```bash
$ python3 preprocess.py SMAP MSL SWaT WADI SMD MSDS UCR MBA NAB
```

To run a model on a dataset, run the following command:
```bash
$ python3 main.py --model <model> --dataset <dataset> --retrain
```
