import numpy as np
from pyqvmbenchmark.unitary_generator import tensor_gates, tensor_up
from pyqvmbenchmark.gates import I
import pyquil.forest as qvm_module
import pyquil.quil as pq
from itertools import product


class Simulator(object):
    """
    Simulator for noisy VQE
    """
    def __init__(self, minimizer):
        """
        Simulator for noisy VQE
        """
        self.minimizer = minimizer
        self.shots = 1
        self.cxn = qvm_module.Connection()
        self.n_qubits = None

    def qve_run_noise_nick(self, parametric_state_evolve, hamiltonian, initial_params,
                           n_qubits,
                           gate_noise=None, measurement_noise=None, dephasing=None,
                           amplitude_damping=None, depolarizing_noise=None,
                           qvm=None, jacobian=None):
        """
        :param depolarizing_noise: probability of applying I X, Y, Z
                                   depolarizing channel after each gate to
                                   every qubit
        """
        # write down the density matrix starting at |0><0| and then go through
        # gate by gate, get_action using get_action_on_basis,
        # propogate_density, cycle

        self.n_qubits = n_qubits
        start_prog = pq.Program()
        for ii in xrange(n_qubits):
            start_prog.inst(I(ii))

        init_wf = self.cxn.wavefunction(start_prog)
        init_wf = init_wf.reshape((-1, 1))
        rho_0 = init_wf.dot(init_wf.T)
        if isinstance(hamiltonian, np.ndarray):
            Htotal = hamiltonian
        else:
            Htotal = tensor_up(hamiltonian.terms, hamiltonian.n_qubits)

        def objective_function(params):
            quil_prog = parametric_state_evolve(params)
            rho = np.copy(rho_0)

            # evolve the density matrix
            for gate in quil_prog.instructions:
                unitary = tensor_gates(gate, self.n_qubits)
                # this is where I'd add some error
                rho = unitary.dot(rho).dot(np.conj(unitary).T)

            # send through depolarizing channel
            if depolarizing_noise is not None:
                rho_depol_channel = np.zeros_like(rho)
                kraus_ops = self.get_depol_unitary(depolarizing_noise)
                for op in kraus_ops:
                    rho_depol_channel += op.dot(rho).dot(np.conj(op).T)
                rho = rho_depol_channel.copy()

            return np.trace(Htotal.dot(rho)).real

        def print_current_iter(iter_vars):
            E = objective_function(iter_vars)
            print("Parameters: {} E => {}".format(iter_vars, E))

        result = self.minimizer(objective_function, initial_params,
                                jac=jacobian,
                                method='BFGS', options={'disp': False},
                                callback=print_current_iter)

        return result.x, result.fun

    def get_dephase_unitary(self, prob_noise):
        dim = 2**self.n_qubits
        I = np.array([[1, 0], [0, 1]])
        Z = np.array([[1, 0], [0, -1]])
        pauli_ops = [I, Z]
        pauli_sets = product(pauli_ops, repeat=self.n_qubits)
        maps = map(lambda x: reduce(np.kron, x[::-1]), pauli_sets)
        coeffs = [np.sqrt((1 - prob_noise) + prob_noise/dim)] + [np.sqrt(prob_noise/float(dim))]*(len(maps)-1)
        maps = map(lambda x: x[0]*x[1], zip(coeffs, maps))

        test_identity = np.zeros((2**self.n_qubits, 2**self.n_qubits),
                                 dtype=complex)
        for op in maps:
            test_identity += np.conj(op).T.dot(op)
        assert np.isclose(np.eye(2**self.n_qubits), test_identity).all()
        return maps

    def get_depol_unitary(self, prob_noise):
        dim = 4**self.n_qubits
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        I = np.array([[1, 0], [0, 1]])
        pauli_ops = [I, X, Y, Z]
        pauli_sets = product(pauli_ops, repeat=self.n_qubits)
        maps = map(lambda x: reduce(np.kron, x[::-1]), pauli_sets)
        coeffs = [np.sqrt((1 - prob_noise) + prob_noise/dim)] + [np.sqrt(prob_noise/float(dim))]*(len(maps)-1)

        maps = map(lambda x: x[0]*x[1], zip(coeffs, maps))

        test_identity = np.zeros((2**self.n_qubits, 2**self.n_qubits),
                                 dtype=complex)
        for op in maps:
            test_identity += np.conj(op).T.dot(op)
        assert np.isclose(np.eye(2**self.n_qubits), test_identity).all()
        return maps
