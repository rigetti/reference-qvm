[![Build Status](https://semaphoreci.com/api/v1/rigetti/reference-qvm/branches/master/badge.svg)](https://semaphoreci.com/rigetti/reference-qvm)

# Reference QVM

The `referenceqvm` is the reference implementation of the QVM outlined in the
arXiv:1608:03355 by Robert Smith, Spike Curtis, and Will Zeng. It is a research package that 
supports rapid prototyping and development of quantum programs using pyQuil.

Currently, this QVM supports a subset of functionality in the Quil specifications, 
excepting certain functions (DEFCIRCUIT, WAIT, NOP).

Noise models (dephasing, Kraus operators), parametrization with bits in 
classical memory, and other features will be added in future releases.


## Installation

You can install reference-qvm directly from the Python package manager `pip` using:
```
pip install referenceqvm
```

To instead install reference-qvm from source, clone this repository, `cd` into it, and run:
```
pip install -r requirements.txt -e .
```

This will install the reference-qvm's dependencies if you do not already have them.

## Development and Testing

We use tox and pytest for testing. Tests can be executed from the top-level directory by simply
running:
```
tox
```
The setup is currently testing Python 2.7 and Python 3.6.

## Building the Docs

We use sphinx to build the documentation. To do this, navigate into pyQuil's top-level directory and run:

```
sphinx-build -b html ./docs/source ./docs/build
```
To view the docs navigate to the newly-created `docs/build` directory and open
the `index.html` file in a browser. Note that we use the Read the Docs theme for
our documentation, so this may need to be installed using `pip install sphinx_rtd_theme`.

## Interaction with the referenceqvm

The qvm can be accessed in a similar way to the Forest QVM access.
Start by importing the synchronous connection object from the `referenceqvm.api` module

```python
from referenceqvm.api import SyncConnection
```

and initialize a connection to the reference-qvm

```python
qvm = SyncConnection()
```

By default, the Connection object uses the wavefunction transition type.  

Then call the `qvm.wavefunction(prog)` method to get back the classical memory and the 
pyquil.Wavefunction object given a pyquil.quil.Program object `prog`.

The reference-qvm has the same functionality as Forest QVM and is useful for testing 
small quantum programs on a local machine.  For example, the same code (up to the 
`referenceqvm.api` import) can be used to simulate pyquil programs.

```python
>>> import pyquil.quil as pq
>>> import referenceqvm.api as api
>>> from pyquil.gates import *
>>> qvm = api.SyncConnection()
>>> p = pq.Program(H(0), CNOT(0,1))
<pyquil.pyquil.Program object at 0x101ebfb50>
>>> qvm.wavefunction(p)[0]
[(0.7071067811865475+0j), 0j, 0j, (0.7071067811865475+0j)]
```

SyncConnection can also initialize a QVM that does not return a wavefunction but instead a unitary corresponding
to the pyquil program.  This can be extremely useful in terms of debugging and understanding gate physics.  For example,
we can examine the unitary for a CNOT operator.

```python
>>> import pyquil.quil as pq
>>> import referenceqvm.api as api
>>> from pyquil.gates import CNOT
>>> qvm = api.SyncConnection(type_trans='unitary')
>>> p = pq.Program(CNOT(1, 0))
>>> u = qvm.unitary(p)
>>> print(u)
[[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]
 [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]]
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
  commit = {the commit you used}
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

