Overview
========

Welcome to reference-qvm!

The reference-qvm is a python implementation of the quantum-abstract-machine described in
arXiv:1608:03355 by Robert Smith, Spike Curtis, and Will Zeng.  The implementation of this quantum-virtual-machine
is meant to be as pedagogical and extensible as possible.  The python code attempts to stay very close to the
mathematical description of the state-machine model of the qvm described in the original paper.  The implementation
goes as far as using the same variable names as the white-paper in the hopes that the implementation is readable and
easy to understand.

The reference-qvm is a quantum simulator of pyquil programs.  It compliments the Forest-QVM, provided by Rigetti
Computing, by providing a easy to install quantum-virtual-machine with the same API's as the Rigetti Forest-QVM accessed
through pyquil.  It runs locally without the need for special keys or access rules.  Ideally, small experiments and
programs written in pyquil and grove can be run with the reference-qvm while larger jobs or performance sensitive
simulations can be run with the Forest-QVM.

The reference-qvm provides some functionality beyond the current Forest-QVM:

1. Getting the unitary corresponding to a pyQuil program
2. Easy customization of error models by specification of pre or post hooks in the state-machine model

Upcoming development roadmap for reference-qvm:

1. Standard class of error models for gates and measurement: T1/T2, bit-, phase-, bit-phase-flip, depolarizing channel
2. Stochastic evolution of the wavefunction under a noise model
3. Density matrix evolution (pure states and mixed states)
