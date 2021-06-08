# fPAKE Benchmark

This folder contains the fPAKE protocol implementation and a benchmarking script for the paper "FastZIP: Faster and More Secure Zero-Interaction Pairing", by Mikhail Fomichev, Julia Hesse, Lars Almon, Timm Lippert, Jun Han, Matthias Hollick, in Proceedings of the 19th Annual International Conference on Mobile Systems, Applications, and Services (MobiSys '21).

## Requirements

- Python 3.7.
- Cryptographic Library for symmetric and asymmetric cryptography: https://cryptography.io/en/latest/.
- Two Raspberry Pi 3 Model devices with the default Raspbian connected to the same network.

## Getting Started

* Copy the "fPAKE" folder to the target devices (e.g., Raspberry Pis).

* Copy the "cache" folder from the "[fastzip-results/fpake/input](https://dx.doi.org/10.5281/zenodo.4911390)" into the "fPAKE" folder and at the same level as *benchmark.py*.

* Configure the *config.ini* on both devices in the "fPAKE" directory as follows:

  * *IP*
    * On the receiving device(-s) IP (e.g., `IP=192.168.0.102`).
    * Set IP to to *localhost* if to run the benchmarking on the same device.
  * *ROLE*
    * A device can either be a sender or receiver (i.e., `ROLE=receiver | ROLE=sender`).
  * *SECPARAM* 
    * Should be set to the same value on both devices, represents a security level (i.e., `SECPARAM=1 | SECPARAM=0`).
  * *JSONDIR*
    - Should point to the directory that contains the input data (i.e., `JSONDIR=.../input/cache`). 

* Execute code:

  *python3 benchmark.py*

* The results should be saved based on the security level in the folder "results128" or "results244" (see "[fastzip-results/fpake/results](https://dx.doi.org/10.5281/zenodo.4911390)").

* With the *statistics.py* (should be placed in the root of "fastzip-results/fpake/results") the results can be further evaluated:

  * Open the *statistics.py* script and set the following: 

    * *default_folder = "results128"* or *default_folder = "results244"*.

  * The statistics results will be printed to the console by executing:

    *python3 statistics.py*


## Authors

Timm Lippert, Julia Hesse, Mikhail Fomichev


## License

The code is licensed under the GNU GPLv3 - see [LICENSE](https://github.com/seemoo-lab/fastzip/blob/main/LICENSE) for details.

