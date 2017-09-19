from pyquil.api import SyncConnection
from pyquil.quil import Program
from pyquil.gates import *
import numpy as np
import pytest
from grove.qft.fourier import qft


@pytest.mark.skip(reason="Must add support for Forest connections in testing")
def tests_against_cloud(qvm):
    """
    """
    qvm_cloud = SyncConnection()

    # simple program
    p = Program(H(0))
    cloud_results = qvm_cloud.run(p, classical_addresses=[0], trials=10)
    local_results = qvm.run(p, classical_addresses=[0], trials=10)
    assert len(cloud_results) == len(local_results)
    
    cloud_wf, _ = qvm_cloud.wavefunction(p)
    local_wf, _ = qvm.wavefunction(p)
    assert np.allclose(cloud_wf.amplitudes.reshape((-1, 1)), \
                       local_wf.amplitudes)

    # complex program
    p = qft(range(16))
    cloud_wf, _ = qvm_cloud.wavefunction(p)
    local_wf, _ = qvm.wavefunction(p)
    assert np.allclose(cloud_wf.amplitudes.reshape((-1, 1)), \
                       local_wf.amplitudes)
