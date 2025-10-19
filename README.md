# Adversarial Representation Learning for Canonical Correlation Analysis

### ICLR 2023 submission

adMDM is a deep learning approach that learns maximally correlated latent representations from multimodal data. This implementation provides source code, simulated data generation tools and demonstrations.

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
adMDM requires three steps to learn canonical representations from multimodal data. 

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
