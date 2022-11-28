# postmetad

`postmetad` is a Python package for post-processing PLUMED [Metadynamics](https://www.plumed.org/doc-v2.8/user-doc/html/_m_e_t_a_d.html) and [On-the-fly Probability Enhanced Sampling (OPES)](https://www.plumed.org/doc-v2.8/user-doc/html/_o_p_e_s.html) simulations. 

`postmetad.metad` can reconstruct biases and free energy profiles from METAD output files, while 
`postmetad.opes_metad` can reconstruct biases and free energy profiles from OPES_METAD output files.

## Installation

Clone the sources with git, and run 
```sh
pip install [-e] .
``` 
at the top level directory of the repository.

## Usage

Read the documentation at [apallath.github.io/postmetad](apallath.github.io/postmetad)

## References

1. Laio, A., & Parrinello, M. (2002). Escaping free-energy minima. Proceedings of the National Academy of Sciences, 99(20), 12562–12566. [DOI](https://doi.org/10.1073/pnas.202427399)
2. Invernizzi, M., & Parrinello, M. (2020). Rethinking metadynamics: From bias potentials to probability distributions. The Journal of Physical Chemistry Letters, 11(7), 2731–2736. [DOI](https://doi.org/10.1021/acs.jpclett.0c00497)




