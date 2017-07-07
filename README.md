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

Interaction with the referenceqvm
---------------------------------

The qvm can be accessed in a similar way to the Forest QVM access.
Start by importing qvm from the `referenceqvm` module

```
from referenceqvm import qvm
```

Then call the `wavefunction()` or `unitary()` method to get the wavefunction
or unitary corresponding to the execution of the input program.
