Reference QVM
-------------

The `referenceqvm` is the reference implementation of the QVM outlined in the
arXiv:1608:03355 by Robert Smith, Spike Curtis, and Will Zeng.  This
implementation currently only support protoQuil (a subset of Quil instructions)
represented as pyQuil programs.


Interaction with the referenceqvm
---------------------------------

The qvm can be accessed in a similar way to the forest QVM access.  
Start by importing qvm from the `referenceqvm` module

```
from referenceqvm import qvm
```

Then call the `wavefunction()` or `density()` method to get the wavefunction or
density back.  

The library can be pip installed as usual.
