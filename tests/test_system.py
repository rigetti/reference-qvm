from pyquil.api import SyncConnection
from pyquil.quil import Program
from pyquil.gates import *
from pyquil.paulis import PauliTerm, PauliSum, exponentiate
import numpy as np
import pytest
from referenceqvm.gates import gate_matrix


def tests_against_cloud(qvm, qvm_unitary):
    """
    """
