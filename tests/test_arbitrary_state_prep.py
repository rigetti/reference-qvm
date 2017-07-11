import pytest
from pyquil.quil import Program
from pyquil.gates import H as Hgate
from pyquil.gates import CNOT as CNOTgate
from pyquil.gates import Y as Ygate
from pyquil.gates import X as Xgate
from pyquil.gates import Z as Zgate
from pyquil.gates import I as Igate
from pyquil.gates import RX as RXgate
from pyquil.gates import RY as RYgate
from pyquil.gates import RZ as RZgate
from pyquil.gates import PHASE as PHASEgate
from pyquil.gates import _make_gate
from pyquil.gates import STANDARD_GATES
from pyquil.paulis import PauliTerm, PauliSum, exponentiate
from referenceqvm.api import SyncConnection
from referenceqvm.qvm import QVM, QVM_Unitary
from referenceqvm.gates import gate_matrix
from grove.arbitrary_state.arbitrary_state import create_arbitrary_state
import numpy as np
from tqdm import tqdm


def test_generate_arbitrary_states():
    qvm = SyncConnection()

    for length in tqdm(range(1, 2 ** 4 + 1)):
        v = 10 * np.random.random() * (np.random.random(length) - 0.5) \
          + 5j * (np.random.random(length) - 0.5)
        norm = np.sqrt(np.sum(np.multiply(np.conj(v), v)))
        p = create_arbitrary_state(v)
        wf, _ = qvm.wavefunction(p)
        # check actual part of wavefunction
        assert np.allclose(v.reshape((-1, 1)), wf.amplitudes[:len(v),:] * norm)
        # check remaining zeros part of wavefunction
        assert np.allclose(np.zeros((wf.amplitudes.shape[0] - len(v), 1)), \
                           wf.amplitudes[len(v):] * norm)


if __name__ == "__main__":
    test_generate_arbitrary_states()
