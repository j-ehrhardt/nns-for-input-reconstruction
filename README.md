[![Python Version](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://www.python.org/downloads/release/python-380/)
[![Mamba](https://img.shields.io/badge/Mamba-1.5.8-green)](https://mamba.readthedocs.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# Using Backpropagation to Reconstruct Missing Inputs of Neural Networks

This repository contains the code for reconstructing missing input data of neural networks via backpropagation. 
The method is inspired by [Roche et al. 2023](https://arxiv.org/abs/2308.10496). 

The idea is that trained weights of a neural network can be used to reconstruct missing input data to the neural network, by simply including the missing data in the gradient and optimizing its values via backpropagation, while fixing the weights of the neural network. 

We extend Roche et al's approach by instantiating multiple neural network instances. This allows us to counter being captured in a local minimum during the optimization.
The most promising candidate is then selected for final optimization. 


## Experimental Evaluation
We evaluate the results with the MNIST dataset.
A full empirical evaluation can be found the publication above.
For our evaluation, we trained a simple Autoencoder on the complete MNIST dataset. 
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
