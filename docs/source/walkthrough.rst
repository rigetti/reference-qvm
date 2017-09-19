Walkthrough
===============================================================================

Here, we detail the operation of several
important functions in reference-qvm, in particular the generation of
unitaries from known matrices (stored in `gates`).

Bitstring Ordering
-------------------------------------------------------------------------------
For :math:`n` qubits, computational basis states are typically ordered in
literature according to the increasing basis order:

.. math::
    \def\ket#1{\mathinner{|{#1}\rangle}}
.. math::
    \ket{\psi} =
    \ket{b_0}_0 \otimes \ket{b_1}_1 \otimes \cdots \otimes \ket{b_{n - 1}}_{n-1}

In reference-qvm and in many applications, it is often more convenient and
concise to label basis states in the decreasing basis order:

.. math::
    \ket{\psi} =
    \ket{b_{n-1} \cdots b_1 b_0}

There are several reasons to do this, the main being semantic ordering of
basis states.

When new qubits are added to the system with basis states ordered by increasing
qubit index (as in Nielsen and Chuang and most quantum information references),
the binary bitstring labeling of the states preserves its ordering when we
*sort* the basis states in increasing order.

For example, the basis state :math:`\ket{110011}` is a basis state of a
:math:`n=6` Hilbert space, and has index :math:`2^5 + 2^4 + 2^1 + 2^0 = 51`.
When we tensor on another Hilbert space of :math:`m=2` qubits, the state in
the new Hilbert space is :math:`\ket{00110011}` - which preserves the exact
same index as before!

Labeling our basis states in decreasing basis order still allows us to
construct the unitary that evolves the full Hilbert space of :math:`n` qubits
from a :math:`k`-qubit gate by simply tensoring up the matrix corresponding to
the :math:`k`-qubit gate with identity matrices on the adjacent single-qubit
Hilbert spaces.

Unitary Generator Functionality
-------------------------------------------------------------------------------
This leads us directly to a discussion of the implementation of qubit evolution
in reference-qvm. `gates` stores the matrix representations of
the standard gate set, including 1-qubit gates (X, Y, Z, etc.), 2-qubit gates
(CNOT, SWAP, etc.), and several 3-qubit gates (CCNOT, CSWAP, etc.).

Suppose a pyQuil program applies a k-qubit gate :math:`\hat{G}` on adjacent
qubits :math:`[j+k-1, \cdots, j+1, j]`. If the k-qubit gate has a matrix
representation which determines how it evolves a k-qubit Hilbert space of
indices :math:`[k-1, \cdots, 1, 0]`, then the n-qubit matrix resulting from
operating the k-qubit gate on the aforementioned qubits is directly formed by
the tensor product of the k-qubit matrix with identities:

.. math::
    \hat{G}_{\text{full}} =
                \underbrace{I_{n-1} \otimes I_{n-2} \otimes \cdots I_{j+k}}_
                           {\text{upper $n-k-j$ qubits}}
        \otimes \underbrace{\hat{G}}_{\text{$k$-qubit operated}} \otimes
                \underbrace{I_{j-1} \otimes \cdots \otimes I_{1} I_0}_
                           {\text{lower $j$ qubits}}

This procedure is termed *lifting* - we are lifting the :math:`\hat{G}` matrix
from a :math:`k`-qubit Hilbert space to the full :math:`n`-qubit Hilbert space
via identity tensoring.

This enables the user of pyQuil/reference-qvm to form an arbitrary k-qubit
gate, as long as the matrix defining its operation is known. It is a simple
procedure to then `DefGate` the gate, and use it in any pyQuil program.

Swapping of Hilbert spaces
-------------------------------------------------------------------------------

In the previous example, it was a simple matter to apply a :math:`k`-qubit gate
operating on **adjacent** Hilbert spaces - just tensor everything up!

In general, we should support arbitrary gates, operating on qubits that are
arbitrarily far apart in terms Hilbert space indices. What if I apply
```CCNOT 5 10 3```?

This is where swapping comes in. `unitary_generator` uses the 2-qubit symmetric
SWAP matrix to permute Hilbert spaces into adjacency, then applies the
:math:`k`-qubit gate in matrix form, then permutes the Hilbert spaces back into
their original positions.

For example, ```CCNOT 5 4 3``` will not incur the SWAP overhead, since the
Hilbert spaces are already adjacent and in the proper order to directly apply
the CCNOT matrix (NOT on qubit 3 controlled by qubits 5, 4).

However, ```CCNOT 3 4 5``` (and similar increasing qubit indexing formats as
suggested in most quantum algorithm references) *will* incur the SWAP overhead,
because the CCNOT gate is stored in `gates` as an :math:`2^3 \times 2^3` matrix,
which applies a NOT on qubit 0 controlled by qubits 2, 1 (assuming decreasing
qubit index). Hence, if we want to do ```CCNOT 3 4 5``` we need to apply a NOT
on qubit 5, controlled by qubits 3 and 4 - we need to permute the Hilbert spaces
:math:`[3, 4, 5] \longrightarrow [5, 4, 3]` before we can lift the CCNOT matrix
to the full Hilbert space.

Thus, it is worthwhile to keep this in mind when designing quantum programs.
Large gates incur an overhead depending on how the qubit arguments are ordered.
For example, a simple heuristic to minimize SWAP distance is to keep control
qubits at higher indices than the controlled qubits (opposite the traditional
notion in many quantum algorithms).

We are currently exploring methods of precompiling an input pyQuil program and
relabeling the qubit arguments in a way that minimizes the SWAP overhead (a huge
problem, especially in programs run on real quantum chips, which have limited
connectivity and incur noise errors as the number of SWAP gates increase).

