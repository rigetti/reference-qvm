"""
Testing the QVM stablizer
"""
import sys
import numpy as np
from itertools import product
from functools import reduce
from pyquil.paulis import sI, sX, sY, sZ, PAULI_COEFF, PAULI_OPS
from pyquil.quil import Program
from pyquil.gates import H, S, CNOT, I
from referenceqvm.qvm_stabilizer import (QVM_Stabilizer, pauli_stabilizer_to_binary_stabilizer,
                                         binary_stabilizer_to_pauli_stabilizer)
from referenceqvm.state_actions import project_stabilized_state
from referenceqvm.api import QVMConnection


pauli_subgroup = [sI, sX, sY, sZ]
five_qubit_code_generators = [sX(0) * sZ(1) * sZ(2) * sX(3) * sI(4),
                              sI(0) * sX(1) * sZ(2) * sZ(3) * sX(4),
                              sX(0) * sI(1) * sX(2) * sZ(3) * sZ(4),
                              sZ(0) * sX(1) * sI(2) * sX(3) * sZ(4)]
bell_stabilizer = [sZ(0) * sZ(1), sX(0) * sX(1)]


def test_initialization():
    """
    Test if upon initialization the correct size tableau is set up
    """
    num_qubits = 4
    qvmstab = QVM_Stabilizer(num_qubits=num_qubits)
    assert qvmstab.num_qubits == num_qubits
    assert qvmstab.tableau.shape == (2 * num_qubits, 2 * num_qubits + 1)

    initial_tableau = np.hstack((np.eye(2 * num_qubits), np.zeros((2 * num_qubits, 1))))
    assert np.allclose(initial_tableau, qvmstab.tableau)

    num_qubits = 6
    qvmstab = QVM_Stabilizer(num_qubits=num_qubits)
    assert qvmstab.num_qubits == num_qubits
    assert qvmstab.tableau.shape == (2 * num_qubits, 2 * num_qubits + 1)

    initial_tableau = np.hstack((np.eye(2 * num_qubits), np.zeros((2 * num_qubits, 1))))
    assert np.allclose(initial_tableau, qvmstab.tableau)


def test_stabilizer_to_matrix_conversion():
    # bitflip code
    stabilizer_matrix = pauli_stabilizer_to_binary_stabilizer(bell_stabilizer)
    true_stabilizer_matrix = np.array([[0, 0, 1, 1, 0],
                                       [1, 1, 0, 0, 0]])
    assert np.allclose(true_stabilizer_matrix, stabilizer_matrix)

    test_stabilizer_list = binary_stabilizer_to_pauli_stabilizer(true_stabilizer_matrix)
    for idx, term in enumerate(test_stabilizer_list):
        assert term == bell_stabilizer[idx]

    #  given some codes convert them to code matrices
    stabilizer_matrix = pauli_stabilizer_to_binary_stabilizer(five_qubit_code_generators)
    true_stabilizer_matrix = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
                                      [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                                      [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                                      [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0]])
    assert np.allclose(true_stabilizer_matrix, stabilizer_matrix)

    test_stabilizer_list = binary_stabilizer_to_pauli_stabilizer(true_stabilizer_matrix)
    for idx, term in enumerate(test_stabilizer_list):
        assert term == five_qubit_code_generators[idx]


def test_row_sum_sub_algorithm_g_test():
    """
    Test the behavior of the subroutine of the rowsum routine

    g(x1, z1, x2, z2) is a function that returns the exponent that $i$ is raised (0, -1, 1)
    when pauli matrices represented by x1z1 and x2z2 are multiplied together.

    Logic table:
        I1 = {x1 = 0, z1 = 0}
        I2 = {x2 = 0, z2 = 0}
        g(I1, I2) = 0

        I1 = {x1 = 0, z1 = 0}
        X2 = {x2 = 1, z2 = 0}
        g(I1, X2) = 0

        I1 = {x1 = 0, z1 = 0}
        Y2 = {x2 = 1, z2 = 1}
        g(I1, X2) = 0

        I1 = {x1 = 0, z1 = 0}
        Z2 = {x2 = 0, z2 = 1}
        g(I1, X2) = 0

        ---------------------

        X1 = {x1 = 1, z1 = 0}
        I2 = {x2 = 0, z2 = 0}
        g(I1, I2) = 0

        X1 = {x1 = 1, z1 = 0}
        X2 = {x2 = 1, z2 = 0}
        g(I1, X2) = 0

        X1 = {x1 = 1, z1 = 0}
        Y2 = {x2 = 1, z2 = 1}
        g(I1, X2) = 1

        X1 = {x1 = 1, z1 = 0}
        Z2 = {x2 = 0, z2 = 1}
        g(I1, X2) = -1

        ---------------------
        etc...
    """
    num_qubits = 4
    qvmstab = QVM_Stabilizer(num_qubits=num_qubits)

    # loop over the one-qubit pauli group and make sure the power of $i$ is correct
    # this test would be better if we stored levi-civta
    pauli_map = {'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1), 'I': (0, 0)}
    for pi, pj in product(PAULI_OPS, repeat=2):
        i_coeff = qvmstab._g_update(pauli_map[pi][0], pauli_map[pi][1], pauli_map[pj][0], pauli_map[pj][1])
        # print(pi, pj, "".join([x for x in [pi, pj]]), PAULI_COEFF["".join([x for x in [pi, pj]])], 1j**i_coeff)
        assert np.isclose(PAULI_COEFF["".join([x for x in [pi, pj]])], 1j**i_coeff)


def test_rowsum_phase_accumulator():
    """
    Test the accuracy of the phase accumulator.  This subroutine keeps track of the power that $i$ is raised to
    when multiplying two PauliTerms together.  PauliTerms are now composed of multi single-qubit Pauli's.

    The formula is phase_accumulator(h, i) = 2 * rh + 2 * ri + \sum_{j}^{n}g(x_{ij}, z_{ij}, x_{hj}, z_{hj}

    The returned value is presented mod 4.

    notice that since r_h indicates +1 or -1 this corresponds to i^{0} or i^{2}.  Given r_{h/i} takes on {0, 1} then
    we need to multiply by 2 get the correct power of $i$ for the Pauli Term to account for multiplication.

    In order to test we will generate random elements of P_{n} and then multiply them together keeping track of the
    $i$ power.  We will then compare this to the phase_accumulator subroutine.

    We have to load in the stabilizer into an empty qvmstab object because the subroutine needs a tableau as a reference
    """
    num_qubits = 2
    pauli_terms = [sX(0) * sX(1), sZ(0) * sZ(1)]
    stab_mat = pauli_stabilizer_to_binary_stabilizer(pauli_terms)
    qvmstab = QVM_Stabilizer(num_qubits=num_qubits)
    qvmstab.tableau[num_qubits:, :] = stab_mat
    exp_on_i = qvmstab._rowsum_phase_accumulator(2, 3)
    assert exp_on_i == 2

    num_qubits = 2
    pauli_terms = [sZ(0) * sI(1), sZ(0) * sZ(1)]
    stab_mat = pauli_stabilizer_to_binary_stabilizer(pauli_terms)
    qvmstab = QVM_Stabilizer(num_qubits=num_qubits)
    qvmstab.tableau[num_qubits:, :] = stab_mat
    exp_on_i = qvmstab._rowsum_phase_accumulator(2, 3)
    assert exp_on_i == 0

    # now try generating random valid elements from 2-qubit group
    for _ in range(100):
        num_qubits = 6
        pauli_terms = []
        for _ in range(num_qubits):
            pauli_terms.append(reduce(lambda x, y: x * y, [pauli_subgroup[x](idx) for idx, x in enumerate(np.random.randint(1, 4, num_qubits))]))
        stab_mat = pauli_stabilizer_to_binary_stabilizer(pauli_terms)
        qvmstab = QVM_Stabilizer(num_qubits=num_qubits)
        qvmstab.tableau[num_qubits:, :] = stab_mat
        p_on_i = qvmstab._rowsum_phase_accumulator(num_qubits, num_qubits + 1)
        coeff_test = pauli_terms[1] * pauli_terms[0]
        assert np.isclose(coeff_test.coefficient, (1j) ** p_on_i)


def test_rowsum_full():
    """
    Test the rowsum subroutine.

    This routine replaces row h with P(i) * P(h) where P is the pauli operator represented by the row
    given by the index.  It uses the summation to accumulate whether the phase is +1 or -1 and then
    xors together the elements of the rows replacing row h
    """
    # now try generating random valid elements from 2-qubit group
    for _ in range(100):
        num_qubits = 6
        pauli_terms = []
        for _ in range(num_qubits):
            pauli_terms.append(reduce(lambda x, y: x * y, [pauli_subgroup[x](idx) for idx, x in enumerate(np.random.randint(1, 4, num_qubits))]))
        try:
            stab_mat = pauli_stabilizer_to_binary_stabilizer(pauli_terms)
        except:
            # we have to do this because I'm not strictly making valid n-qubit
            # stabilizers
            continue
        qvmstab = QVM_Stabilizer(num_qubits=num_qubits)
        qvmstab.tableau[num_qubits:, :] = stab_mat
        p_on_i = qvmstab._rowsum_phase_accumulator(num_qubits, num_qubits + 1)
        if p_on_i not in [1, 3]:
            coeff_test = pauli_terms[1] * pauli_terms[0]
            assert np.isclose(coeff_test.coefficient, (1j) ** p_on_i)
            qvmstab._rowsum(num_qubits, num_qubits + 1)
            try:
                pauli_op = binary_stabilizer_to_pauli_stabilizer(qvmstab.tableau[[num_qubits], :])[0]
            except:
                continue
            true_pauli_op = pauli_terms[1] * pauli_terms[0]
            assert pauli_op == true_pauli_op


def test_simulation_hadamard():
    """
    Test if Hadamard is applied correctly to the tableau

    The first example will be a 1 qubit example. where we perform H | 0 > = |0> + |1>.
    Therefore we expect the tableau to represent the stablizer X
    """
    prog = Program().inst(H(0))
    qvmstab = QVM_Stabilizer(num_qubits=1)
    qvmstab._apply_hadamard(prog.instructions[0])
    x_stabilizer = np.array([[0, 1, 0],
                             [1, 0, 0]])
    assert np.allclose(x_stabilizer, qvmstab.tableau)

    qvmstab = QVM_Stabilizer(num_qubits=2)
    qvmstab._apply_hadamard(prog.instructions[0])
    x_stabilizer = np.array([[0, 0, 1, 0, 0],
                             [0, 1, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0]])
    assert np.allclose(x_stabilizer, qvmstab.tableau)

    prog = Program().inst(H(1))
    qvmstab = QVM_Stabilizer(num_qubits=2)
    qvmstab._apply_hadamard(prog.instructions[0])
    x_stabilizer = np.array([[1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 1, 0, 0],
                             [0, 1, 0, 0, 0]])
    assert np.allclose(x_stabilizer, qvmstab.tableau)

    prog = Program().inst([H(0), H(1)])
    qvmstab = QVM_Stabilizer(num_qubits=2)
    qvmstab._apply_hadamard(prog.instructions[0])
    qvmstab._apply_hadamard(prog.instructions[1])
    x_stabilizer = np.array([[0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0],
                             [1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0]])
    assert np.allclose(x_stabilizer, qvmstab.tableau)

    prog = Program().inst([H(0), H(1), H(0), H(1)])
    qvmstab = QVM_Stabilizer(num_qubits=2)
    qvmstab._apply_hadamard(prog.instructions[0])
    qvmstab._apply_hadamard(prog.instructions[1])
    qvmstab._apply_hadamard(prog.instructions[2])
    qvmstab._apply_hadamard(prog.instructions[3])
    x_stabilizer = np.array([[1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0]])
    assert np.allclose(x_stabilizer, qvmstab.tableau)


def test_simulation_phase():
    """
    Test if the Phase gate is applied correctly to the tableau

    S|0> = |0>

    S|+> = |R>
    """
    prog = Program().inst(S(0))
    qvmstab = QVM_Stabilizer(num_qubits=1)
    qvmstab._apply_phase(prog.instructions[0])
    true_stab = np.array([[1, 1, 0],
                          [0, 1, 0]])
    assert np.allclose(true_stab, qvmstab.tableau)

    prog = Program().inst([H(0), S(0)])
    qvmstab = QVM_Stabilizer(num_qubits=1)
    qvmstab._apply_hadamard(prog.instructions[0])
    qvmstab._apply_phase(prog.instructions[1])
    true_stab = np.array([[0, 1, 0],
                          [1, 1, 0]])
    assert np.allclose(true_stab, qvmstab.tableau)


def test_simulation_cnot():
    """
    Test if the simulation of CNOT is accurate
    :return:
    """
    prog = Program().inst([H(0), CNOT(0, 1)])
    qvmstab = QVM_Stabilizer(num_qubits=2)
    qvmstab._apply_hadamard(prog.instructions[0])
    qvmstab._apply_cnot(prog.instructions[1])

    # assert that ZZ XX stabilizes a bell state
    true_stabilizers = [sX(0) * sX(1), sZ(0) * sZ(1)]
    test_paulis = binary_stabilizer_to_pauli_stabilizer(qvmstab.tableau[2:, :])
    for idx, term in enumerate(test_paulis):
        assert term == true_stabilizers[idx]

    # test that CNOT does nothing to |00> state
    prog = Program().inst([CNOT(0, 1)])
    qvmstab = QVM_Stabilizer(num_qubits=2)
    qvmstab._apply_cnot(prog.instructions[0])
    true_tableau = np.array([[1, 1, 0, 0, 0],   # X1 -> X1 X2
                             [0, 1, 0, 0, 0],   # X2 -> X2
                             [0, 0, 1, 0, 0],   # Z1 -> Z1
                             [0, 0, 1, 1, 0]])  # Z2 -> Z1 Z2

    # note that Z1, Z1 Z2 still stabilizees |00>
    assert np.allclose(true_tableau, qvmstab.tableau)


    # test that CNOT produces 11 state after X
    prog = Program().inst([H(0), S(0), S(0), H(0), CNOT(0, 1)])
    qvmstab = QVM_Stabilizer(num_qubits=2)
    qvmstab._apply_hadamard(prog.instructions[0])
    qvmstab._apply_phase(prog.instructions[1])
    qvmstab._apply_phase(prog.instructions[2])
    qvmstab._apply_hadamard(prog.instructions[3])
    qvmstab._apply_cnot(prog.instructions[4])
    true_tableau = np.array([[1, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 1],
                             [0, 0, 1, 1, 0]])

    # note that -Z1, Z1 Z2 still stabilizees |11>
    assert np.allclose(true_tableau, qvmstab.tableau)

    test_paulis = binary_stabilizer_to_pauli_stabilizer(qvmstab.stabilizer_tableau())
    state = project_stabilized_state(test_paulis, qvmstab.num_qubits, classical_state=[1, 1])
    state_2 = project_stabilized_state(test_paulis, qvmstab.num_qubits)

    assert np.allclose(np.array(state.todense()), np.array(state_2.todense()))


def test_measurement_noncommuting():
    """
    Test measurements of stabilizer state from tableau

    This tests the non-commuting measurement operator
    """
    # generate bell state then measure each qubit sequentially
    prog = Program().inst(H(0)).measure(0, 0)
    qvmstab = QVM_Stabilizer()
    results = qvmstab.run(prog, trials=5000)
    assert np.isclose(np.mean(results), 0.5, rtol=0.1)


def test_measurement_commuting():
    """
    Test measuremt of stabilzier state from tableau

    This tests when the measurement operator commutes with the stabilizers

    To test we will first test draw's from a blank stabilizer and then with X
    """
    identity_program = Program().inst([I(0)]).measure(0, 0)
    qvmstab = QVM_Stabilizer()
    results = qvmstab.run(identity_program, trials=1000)
    assert all(np.array(results) == 0)


def test_measurement_commuting_result_one():
    """
    Test measuremt of stabilzier state from tableau

    This tests when the measurement operator commutes with the stabilizers

    This time we will generate the stabilizer -Z|1> = |1> so we we need to do
    a bitflip...not just identity.  A Bitflip is HSSH = X
    """
    identity_program = Program().inst([H(0), S(0), S(0), H(0)]).measure(0, 0)
    qvmstab = QVM_Stabilizer()
    results = qvmstab.run(identity_program, trials=1000)
    assert all(np.array(results) == 1)


def test_bell_state_measurements():
    prog = Program().inst(H(0), CNOT(0, 1)).measure(0, 0).measure(1, 1)
    qvmstab = QVM_Stabilizer()
    results = qvmstab.run(prog, trials=5000)
    assert np.isclose(np.mean(results), 0.5, rtol=0.1)
    assert all([x[0] == x[1] for x in results])


def test_random_stabilizer_circuit():
    """
    Generate a random stabilizer circuit from {CNOT, H, S, MEASURE}

    Compare the outcome to full state evolution
    """
    num_qubits = 2
    # for each qubit pick a set of operations
    gate_operations = {1: H, 2: S, 3: CNOT}
    np.random.seed(42)
    num_gates = 1000 # program has 100 gates in it

    prog = Program()
    for _ in range(num_gates):
        for jj in range(num_qubits):
            instruction_idx = np.random.randint(1, 4)
            if instruction_idx == 3:
                gate = gate_operations[instruction_idx](jj, (jj + 1) % num_qubits)
            else:
                gate = gate_operations[instruction_idx](jj)
            prog.inst(gate)

    qvmstab = QVM_Stabilizer()
    wf = qvmstab.wavefunction(prog)
    wf = wf.amplitudes.reshape((-1, 1))
    rho_trial = wf.dot(np.conj(wf.T))

    qvm = QVMConnection(type_trans='wavefunction')
    rho, _ = qvm.wavefunction(prog)
    rho = rho.amplitudes.reshape((-1, 1))
    rho_true = rho.dot(np.conj(rho.T))
    assert np.allclose(rho_true, rho_trial)

    rho_from_stab = qvmstab.density(prog)
    assert np.allclose(rho_from_stab, rho_true)


if __name__ == "__main__":
    # test_initialization()
    # test_row_sum_sub_algorithm_g_test()
    # test_stabilizer_to_matrix_conversion()
    # test_rowsum_phase_accumulator()
    # test_rowsum_full()
    # test_simulation_hadamard()
    # test_simulation_phase()
    # test_simulation_cnot()
    # test_measurement_noncommuting()
    # test_measurement_commuting()
    # test_measurement_commuting_result_one()
    # test_bell_state_measurements()
    test_random_stabilizer_circuit()
