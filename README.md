# FeatureCollapsing

This repository holds the code associated with the paper: Isaac Sebenius, Topi Paananen, Aki Vehtari. “Feature Collapsing for Gaussian Process Variable Ranking.” _AISTATS_ 2022.

This code uses the GPy framework (https://gpy.readthedocs.io/en/deploy/) for Gaussian Process implementation. The structure of this implementation draws significantly on the code available at https://github.com/topipa/gp-varsel-kl-var corresponding to the paper: Paananen, T., Piironen, J., Andersen, M.R., and Vehtari, A. (2019). Variable selection for Gaussian processes via sensitivity analysis of the posterior predictive distribution. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 89: 1743-1752.

Files:

-FeatureCollapsing-demo.ipynb: shows how FC can easily be used in GP estimation pipeline, recreates results on the toy dataset from the main paper.

-helpers.py: contains the implementation of the FC method as well as other helper functions.

-GP_varsel.py: taken directly from https://github.com/topipa/gp-varsel-kl-var, used to implement the VAR comparison method.
