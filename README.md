# A binned likelihood for stochastic models.
This repository contains example implementations of likelihoods described in [J. High Energ. Phys. (2019) 2019: 30.](https://doi.org/10.1007/JHEP06(2019)030) in both C++ and python.
These implementations are written as stand-alone files so they can be inserted directly into analysis code.

The likelihoods account for the statistical uncertainty inherent when using limited simulation to estimate event rates or expected counts. Advantages of this method include improved coverage properties with respect to other methods, a simple analytic form, and computational performance comparable to the Poisson likelihood. We recommend the likelihood named LEff as a drop-in replacement for the Poisson likelihood in binned analyses.

Citation
--------

If you use this work please cite it as

Argüelles, C.A., Schneider, A. & Yuan, T. J. High Energ. Phys. (2019) 2019: 30.

https://doi.org/10.1007/JHEP06(2019)030

