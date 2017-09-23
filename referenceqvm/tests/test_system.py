from pyquil.quil import Program
from pyquil.gates import *
import numpy as np
from referenceqvm.tests.test_data import data_containers as dc


def tests_against_cloud(qvm):

    # simple program
    p = Program(H(0))
    cloud_results = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
    local_results = qvm.run(p, classical_addresses=[0], trials=10)
    assert len(cloud_results) == len(local_results)
    
    cloud_wf = dc.HADAMARD_WF
    local_wf, _ = qvm.wavefunction(p)
    assert np.allclose(cloud_wf, local_wf.amplitudes)

    # complex program
    p = Program(dc.QFT_8_INSTRUCTIONS)
    cloud_wf = dc.QFT_8_WF_PROBS
    local_wf, _ = qvm.wavefunction(p)
    assert np.allclose(cloud_wf, local_wf.amplitudes)
