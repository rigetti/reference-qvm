import numpy as np
from referenceqvm.tests.test_data import data_containers as dc
from pyquil.quil import Program


def test_generate_arbitrary_states(qvm):
    for k in list(dc.ARBITRARY_STATE_GEN_INSTRUCTIONS.keys()):

        v = np.asarray(dc.ARBITRARY_STATE_GEN_WF[k])
        norm = np.sqrt(np.sum(np.multiply(np.conj(v), v)))

        p = Program(dc.ARBITRARY_STATE_GEN_INSTRUCTIONS[k])
        wf, _ = qvm.wavefunction(p)

        # check actual part of wavefunction
        assert np.allclose(v.reshape((-1, 1)), wf.amplitudes[:len(v), :] * norm)

        # check remaining zeros part of wavefunction
        assert np.allclose(np.zeros((wf.amplitudes.shape[0] - len(v), 1)),
                           wf.amplitudes[len(v):] * norm)
