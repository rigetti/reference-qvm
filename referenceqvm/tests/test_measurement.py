import numpy as np
from pyquil.quil import Program
from pyquil.gates import *


def test_measurement(qvm):
    p = Program()
    p.inst(X(0))
    results = qvm.run_and_measure(p)
    assert len(results) == 1
    assert len(results[0]) == 0

    results = qvm.run_and_measure(p, qubits=[0, 1, 2, 3])
    assert len(results) == 1  # trials=1
    assert len(results[0]) == 4  # four measured qubits
    assert np.allclose(results, [1, 0, 0, 0])


def test_many_sample_measurements(qvm):
    prog = Program().inst(H(0)).measure(0, [0]).measure(0, [1])
    samples = 1000
    result = qvm.run(prog, [0, 1], trials=samples)
    assert all(map(lambda x: x[0] == x[1], result))
    bias = sum(map(lambda x: x[0], result)) / float(samples)
    assert np.isclose(0.5, bias, atol=0.05, rtol=0.05)


def test_biased_coin(qvm):
    # sample from a 75% heads and 25% tails coin
    prog = Program().inst(RX(np.pi / 3, 0))
    samples = 1000
    results = qvm.run_and_measure(prog, qubits=[0], trials=samples)
    coin_bias = sum(map(lambda x: x[0], results)) / float(samples)
    assert np.isclose(coin_bias, 0.25, atol=0.05, rtol=0.05)


