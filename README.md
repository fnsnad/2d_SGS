# Moment Expansion model for non-linear problems in JAX-CFD (Computational Fluid Dynamics in JAX)

Authors: 

JAX-CFD is an experimental research project for exploring the potential of
machine learning, automatic differentiation and hardware accelerators (GPU/TPU)
for computational fluid dynamics. It is implemented in
[JAX](https://github.com/google/jax).

## Getting started

The "notebooks" directory contains the code utilized for the benchamarking and demonstrations of our turbulent flow models.


## Organization

The structure of the libary is the same as in JAX-CFD with the addition of the scripts to reproduce results in
paper ... and it



## Citation

If you use our spectral code, also cite the original JAX-CFD work:

```
@article{Dresdner2022-Spectral-ML,
  doi = {10.48550/ARXIV.2207.00556},
  url = {https://arxiv.org/abs/2207.00556},
  author = {Dresdner, Gideon and Kochkov, Dmitrii and Norgaard, Peter and Zepeda-Núñez, Leonardo and Smith, Jamie A. and Brenner, Michael P. and Hoyer, Stephan},
  title = {Learning to correct spectral methods for simulating turbulent flows},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Local development

To locally install for development:
```
git clone https://github.com/google/jax-cfd.git
cd jax-cfd
pip install jaxlib
pip install -e ".[complete]"
```

Then to manually run the test suite:
```
pytest -n auto jax_cfd --dist=loadfile --ignore=jax_cfd/base/validation_test.py
```
