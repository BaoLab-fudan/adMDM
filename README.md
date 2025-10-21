# Multi-modal Integration with Adversarial Mutual Distribution Matching (adMDM)

adMDM is a deep learning framework for multimodal data integration that simultaneously aligns instance-level and distribution-level representations through mutual adversarial learning.
This repository provides the source code, data simulation tools, and example notebooks used in the study “Multi-modal Integration with Adversarial Mutual Distribution Matching” 

## Package dependency
The software was tested under the following package version

- python == 3.7
- torch == 1.10.0.dev20210607+cu102
- numpy == 1.16.2
- matplotlib == 3.5.3
- umap-learn  == 0.5.1
- pandas == 1.2.4
- seaborn == 0.11.1

## Run adMDM
adMDM learns a shared latent representation across modalities in three steps:

> Step 1: construct adMDM object


```python
mv = adMDM(x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, device=device)
```

where `x_dim` and `y_dim` are input feature dimensions from two modalities, respectively; `z_dim` is the desired  dimension for latent representation.

> Step 2: training adMDM with batch data loader
 
```python
losses_ax, losses_gx, losses_d = mv.fit(train_dl, 
                                        lr=1e-3,
                                        epochs_ae=500,
                                        epochs_inner=500,
                                        epoch_ad=3)
```
where `train_dl` is a standard torch data loader to feed data in batches;  `lr` is the learning rate; `epochs_ae`, `epochs_inner` and `epochs_inner` are epochs of initial autoencoder training, within x-step/y-step training and iterative steps between x-step and y-step.

> Step 3: inference final representations on trained model.


```python
latent_x, latent_y = mv.inference(feature_x.to(device), feature_y.to(device))
```
where `feature_x` and `feature_y` are two modal data used to infer corresponding representations.

## Simulation tool and demonstration

In subdirectory `./simulation/multimodal_simulation.py` provided the function to generate multimodal simulation data with known sample classes. 

`demo.ipynb` provides a demonstration of 1) generation simulated data; 2) representation learning using adMDM from simulated data and 3) visualization of learned representation.


# License

This project is licensed under the MIT License.

MIT License

Copyright (c) 2025 BaoLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
