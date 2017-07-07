"""
Testing sampling of a density matrix
"""
from api import Connection
from pyquil.quil import Program
from pyquil.gates import H, CNOT, RX
import numpy as np
from itertools import product


def test_sample_coin():
    # Flip a coin with a density matrix.
    prog = Program().inst(H(0))
    qvm = Connection(type_trans='wavefunction')

    samples = 10000
    results = qvm.run_and_measure(prog, [0], trials=samples)
    coin_bias = sum(map(lambda x: x[0], results))/float(samples)
    assert np.isclose(coin_bias, 0.5, atol=0.05, rtol=0.05)


def test_sample_bell():
    # Sample a bell state
    prog = Program().inst(H(0), CNOT(0, 1))
    qvm = Connection(type_trans='wavefunction')

    samples = 100000
    results = qvm.run_and_measure(prog, [0, 1], trials=samples)
    for rr in results:
        assert rr[0] == rr[1]

    bias_pair = sum(map(lambda x: x[0], results))/float(samples)
    assert np.isclose(bias_pair, 0.5, atol=0.05, rtol=0.05)


def test_biased_coin():
    # sample from a %75 head and 25% tails coin
    prog = Program().inst(RX(np.pi/3)(0))
    qvm = Connection(type_trans='wavefunction')
    samples = 100000
    results = qvm.run_and_measure(prog, [0], trials=samples)
    coin_bias = sum(map(lambda x: x[0], results))/float(samples)
    assert np.isclose(coin_bias, 0.25, atol=0.05, rtol=0.05)


def test_measurement():
    qvm = Connection(type_trans='wavefunction')
    prog = Program().inst(H(0)).measure(0, [0]).measure(0, [1])
    samples = 10000
    result = qvm.run(prog, [0, 1], trials=samples)
    assert all(map(lambda x: x[0] == x[1], result))
    bias = sum(map(lambda x: x[0], result))/float(samples)
    assert np.isclose(0.5, bias, atol=0.05, rtol=0.05)


def test_marginalization_order():
    samples = 10000
    def gen_rho():
        a = np.random.random()
        b = np.random.random()
        c = np.random.random()
        d = np.random.random()
        H = np.array([[a, b], [c, d]])
        rho = np.dot(H, H.T)/np.trace(np.dot(H, H.T))
        return rho

    def fast_sample(rho, trials=1):
        cdf = np.cumsum(np.diag(rho))
        num_qubits = int(np.log2(np.shape(rho)[0]))
        bit_results = []
        for _ in xrange(trials):
            u = np.random.random()
            state_index = np.searchsorted(cdf, u)
            if state_index == len(cdf):
                state_index = state_index - 1
            bit_results.append(map(int,
                                   np.binary_repr(state_index, width=num_qubits)))

        return bit_results

    rho_0 = gen_rho()
    rho_1 = gen_rho()
    rho = np.kron(rho_1, rho_0)
    rhot = np.zeros((2, 2, 2, 2))
    for i, j, k, l in product(range(2), repeat=4):
        rhot[i, j, k, l] = rho[i + 2*j, k + 2*l]
    assert np.isclose(np.einsum('ijij', rhot), 1.0)

    # contract the lowest index
    assert np.isclose(np.einsum('kikj->ij', rhot), rho_1).all()

    # contract the second lowest index
    assert np.isclose(np.einsum('ikjk->ij', rhot), rho_0).all()

    result = fast_sample(rho, trials=samples)
    bias_q0_1 = sum(map(lambda x: x[0], result))/float(samples)
    assert np.isclose(bias_q0_1, rho_1[1, 1], atol=0.05, rtol=0.05)


if __name__ == "__main__":
    test_sample_coin()
    test_sample_bell()
    test_biased_coin()
    test_measurement()
    test_marginalization_order()
