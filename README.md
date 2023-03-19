# FooBaR
This repository contains an implementation of the [FooBaR attack](https://arxiv.org/abs/2109.11249) on various deep neural network architectures.

Original FooBar attack implementation can be found at https://github.com/martin-ochoa/foobar.

## What's new
- testing some of the popular DNN architectures, such as ResNet
- Implementing fault attack simulation during training phase using GPU acceleration (PyTorch)
- Using gradient descent algorithm for generation of fooling images
- Ability for fault networks in later stages of forward propagation
- Validation of the fooling inputs against non-compromised networks


