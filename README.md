Reference QVM
-------------

The `referenceqvm` is the reference implementation of the QVM outlined in the
arXiv:1608:03355 by Robert Smith, Spike Curtis, and Will Zeng. The purpose of
this rQVM is to allow rapid prototyping and development of quantum programs
using pyQuil.

Currently, this QVM supports all functionality in the Quil specifications, 
excepting certain functions (DEFCIRCUIT, WAIT, NOP).

Noise models (dephasing, Kraus operators), parametrization with bits in 
classical memory, and other features will be added soon.


Installation
------------

You can install reference-qvm directly from the Python package manager `pip` using:
```
pip install reference-qvm
```

To instead install reference-qvm from source, clone this repository, `cd` into it, and run:
```
pip install -r requirements.txt -e .
```

This will install the reference-qvm's dependencies if you do not already have them.

Dependencies
------------

* NumPy
* SciPy
* pyquil
* tqdm (optional, for development testing)
* pytest (optional, for development testing)
* Grove (optional, for development testing)


Interaction with the referenceqvm
---------------------------------

The qvm can be accessed in a similar way to the Forest QVM access.
Start by importing qvm from the `referenceqvm` module

```
from referenceqvm import qvm
```

Then call the `wavefunction()` or `unitary()` method to get the wavefunction
or unitary corresponding to the execution of the input program.


Development and Testing
-----------------------

We use pytest for testing. Tests can be run from the top-level directory using:
```
py.test
```

## How to cite the reference-qvm

If you use the reference-qvm please cite the repository as follows:

bibTex:
```
@misc{rqvm2017.0.0.1,
  author = {Rigetti Computing",
  title = {Reference-QVM},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rigetticomputing},
  commit = {0}
}
```

and the paper outlining the Mathematical specification of the quantum-abstract-machine:

bibTeX:
```
@misc{1608.03355,
  title={A Practical Quantum Instruction Set Architecture},
  author={Smith, Robert S and Curtis, Michael J and Zeng, William J},
  journal={arXiv preprint arXiv:1608.03355},
  year={2016}
}
```

