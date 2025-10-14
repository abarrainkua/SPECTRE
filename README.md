# SPECTRE

This repository contains the code to run SPECTRE, a framework to enhance minimax fairness guarantees without explicit access to demographic information. 

This approach builds upon the structure of the [Minimax Risk Classifier](http://www.jmlr.org/papers/v24/22-0339.html), extending it with additional specifications 
designed to promote fairness guarantees without requiring demographic information. Specifically, we map the original data into the frequency domain using a simple 
Fourier feature mapping, adjust the spectral components to encourage minimax fairness, and constrain the divergence between the worst-case and empirical distributions 
to prevent excessive pessimism and undue sensitivity to outliers.

The implementation of the MRC is based on its original code. 

Currently, the available code supports the following datasets:
- American Community Survey datasets  (through the `folktables` package in python)
- COMPAS
- Toy dataset

However, the implementation of new datasets is straightforward.
