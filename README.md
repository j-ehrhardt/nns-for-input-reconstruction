![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

# Using Backpropagation to Reconstruct Missing Inputs of Neural Networks

This repository contains the code for reconstructing missing input data to neural networks with backpropagation. 
The method is inspired from [Roche et al. 2023](https://arxiv.org/abs/2308.10496). 

The idea is that the trained weights of a neural network can be used to reconstruct missing input data to the neural network, 
by simply including the missing data in the gradient and optimize its values via backpropagation.

## Experimental Evaluation
We evaluate the results with the MNIST dataset (a full empirical evaluation can be found the publication above)
Therefore, we trained a simple Autoencoder on the complete MNIST dataset. 
We then masked single samples with a zero-tensor, as missing values.

**Sample & Masked Sample**
<p>
  <img src="figures/original.png" alt="sample" style="width: 200px; display: inline-block; margin-right: 10px;"/>
  <img src="figures/masked.png" alt="masked sample" style="width: 200px; display: inline-block; margin-right: 10px;"/>
</p>

We included only the missing values as optimizable parameters in the optimizer and optimized it over $n$ epochs.
Reconstructions on during different iterations.

**Reconstruction of the masked sample over different iterations of optimization**
<p>
  <img src="figures/rec1.png" alt="reconstruction 1" style="width: 200px; display: inline-block; margin-right: 10px;"/>
  <img src="figures/rec2.png" alt="reconstruction 2" style="width: 200px; display: inline-block; margin-right: 10px;"/>
  <img src="figures/recn.png" alt="reconstruction n" style="width: 200px; display: inline-block;"/>
</p>

## Requirements

`python >= 3.8`
`torch >= 2`
`numpy >= 1.23`
`matplotlib.pyplot >= 3.8`

## LICENSE 

Licensed under MIT license
