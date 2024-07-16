# Repository for Studying Interpretable DL

This repo serves as a work in progress for studying Interpretability of Deep Learning algorithms. Initially, we will be implementing the Layer-wise Relevance Propagation algorithms using PyTorch for building our models. The models we are interested in are for imaging and for sequence based approaches, as we are usually working with strings of characters and medical imaging. 

## Initial model: 

We will start by analyzing with a benchmark dataset, which is MNIST. We will build both classifiers and autoencoders as: 

- Fully Connected Networks (Multilayer Perceptrons), for Classification tasks. 
- Convolutional Neural Networks (CNN), for Classification tasks.
- Autoencoder with Convolutional layers, for dimensionality reduction/compression/projection tasks.
- Denoising Autoencoder, similar to the previous one. 

We are aiming to show as well how to do Hyperparameter tuning on these models, and then check the LRP algorithms to check how they are working. To study further the LRP, we will be using as a reference:

- [This paper from FHH Institute, in Germany.](https://iphome.hhi.de/samek/pdf/MonXAI19.pdf)
- This list is a work in progress. 

## To do: 

- We need to create functions for training or define a `fit` method in the encoder model. It would be also helpful to create an `encoders.py` file to call.
- In such file, we should put both the denoising autoencoder and the vanilla autoencoder.