FastZIP Codebase
==================================

This repository contains the codebase for the FastZIP zero-interaction pairing scheme published as: "FastZIP: Faster and More Secure Zero-Interaction Pairing" by Mikhail Fomichev, Julia Hesse, Lars Almon, Timm Lippert, Jun Han, Matthias Hollick in *Proceedings of the 19th Annual International Conference on Mobile Systems, Applications, and Services (MobiSys 2021)*.

The relevant datasets can be found [on Zenodo](https://doi.org/10.5281/zenodo.4777836). The pre-print version of our paper is available on [arXiv](https://arxiv.org/abs/2106.04907).

The code in this repository is structured in several folders:

- **Scheme** contains the core functionality of the FastZIP scheme. 
- **Entropy-evaluation** contains instructions and the version of NIST SP 800 90B tests that we used to evaluate min-entropy of our fingerprints. 
- **fPAKE-benchmark** contains the implementation of the fPAKE protocol and scripts to perform its benchmarking. 

# License
All code is licensed under the GNU GPLv3, unless noted otherwise. See LICENSE for details.
