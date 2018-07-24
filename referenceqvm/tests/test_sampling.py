"""
Testing sampling of a density matrix
"""
import numpy as np
from itertools import product

from referenceqvm.api import QVMConnection
from pyquil.quil import Program
from pyquil.gates import H, CNOT, RX


def test_sample_coin():
    # Flip a coin with a density matrix.
    prog = Program().inst(H(0))
    qvm = QVMConnection(type_trans='density')

    samples = 10000
    results = qvm.run_and_measure(prog, trials=samples)
    coin_bias = sum(map(lambda x: x[0], results))/float(samples)
    assert np.isclose(coin_bias, 0.5, atol=0.05, rtol=0.05)


def test_sample_bell():
    # Sample a bell state
    prog = Program().inst(H(0), CNOT(0, 1))
    qvm = QVMConnection(type_trans='density')

    samples = 100000
    results = qvm.run_and_measure(prog, trials=samples)
    for rr in results:
        assert rr[0] == rr[1]

    bias_pair = sum(map(lambda x: x[0], results))/float(samples)
    assert np.isclose(bias_pair, 0.5, atol=0.05, rtol=0.05)


def test_biased_coin():
    # sample from a %75 head and 25% tails coin
    prog = Program().inst(RX(np.pi/3, 0))
    qvm = QVMConnection(type_trans='density')
    samples = 100000
    results = qvm.run_and_measure(prog, trials=samples)
    coin_bias = sum(map(lambda x: x[0], results))/float(samples)
    assert np.isclose(coin_bias, 0.25, atol=0.05, rtol=0.05)


def test_measurement():
    qvm = QVMConnection(type_trans='density')
    prog = Program().inst(H(0)).measure(0, [0]).measure(0, [1])
    samples = 10000
    result = qvm.run(prog, [0, 1], trials=samples)
    assert all(map(lambda x: x[0] == x[1], result))
    bias = sum(map(lambda x: x[0], result))/float(samples)
    assert np.isclose(0.5, bias, atol=0.05, rtol=0.05)
