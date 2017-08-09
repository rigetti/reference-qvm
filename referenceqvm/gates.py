"""
Standard gate set, as detailed in Quil whitepaper (arXiV:1608:03355v2)
"""
import numpy as np
import cmath

I = np.array([[1.0, 0.0], [0.0, 1.0]])

X = np.array([[0.0, 1.0], [1.0, 0.0]])

Y = np.array([[0.0, 0.0 - 1.0j], [0.0 + 1.0j, 0.0]])

Z = np.array([[1.0, 0.0], [0.0, -1.0]])

H = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]])

S = np.array([[1.0, 0.0], [0.0, 1.0j]])

T = np.array([[1.0, 0.0], [0.0, cmath.exp(1.0j * np.pi / 4.0)]])


def PHASE(phi):
    return np.array([[1.0, 0.0], [0.0, np.exp(1j * phi)]])


def RX(phi):
    return np.array([[np.cos(phi / 2.0), -1j * np.sin(phi / 2.0)],
                     [-1j * np.sin(phi / 2.0), np.cos(phi / 2.0)]])


def RY(phi):
    return np.array([[np.cos(phi / 2.0), -np.sin(phi / 2.0)],
                     [np.sin(phi / 2.0), np.cos(phi / 2.0)]])


def RZ(phi):
    return np.array([[np.cos(phi / 2.0) - 1j * np.sin(phi / 2.0), 0],
                     [0, np.cos(phi / 2.0) + 1j * np.sin(phi / 2.0)]])


CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

CCNOT = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 1, 0]])


def CPHASE00(phi):
    return np.diag([np.exp(1j * phi), 1.0, 1.0, 1.0])


def CPHASE01(phi):
    return np.diag([1.0, np.exp(1j * phi), 1.0, 1.0])


def CPHASE10(phi):
    return np.diag([1.0, 1.0, np.exp(1j * phi), 1.0])


def CPHASE(phi):
    return np.diag([1.0, 1.0, 1.0, np.exp(1j * phi)])


SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])

CSWAP = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1]])

ISWAP = np.array([[1, 0, 0, 0],
                  [0, 0, 1j, 0],
                  [0, 1j, 0, 0],
                  [0, 0, 0, 1]])


def PSWAP(phi):
    return np.array([[1, 0, 0, 0],
                     [0, 0, np.exp(1j * phi), 0],
                     [0, np.exp(1j * phi), 0, 0],
                     [0, 0, 0, 1]])


"""
Specialized useful gates; not officially in standard gate set
"""


def BARENCO(alpha, phi, theta):
    lower_unitary = np.array([
        [np.exp(1j * phi) * np.cos(theta), \
         -1j * np.exp(1j * (alpha - phi)) * np.sin(theta)],
        [-1j * np.exp(1j * (alpha + phi)) * np.sin(theta), \
         np.exp(1j * alpha) * np.cos(theta)]])
    return np.diag(np.eye(2), lower_unitary)


gate_matrix = {'I': I,
               'X': X,
               'Y': Y,
               'Z': Z,
               'H': H,
               'S': S,
               'T': T,
               'PHASE': PHASE,
               'RX': RX,
               'RY': RY,
               'RZ': RZ,
               'CNOT': CNOT,
               'CCNOT': CCNOT,
               'CPHASE00': CPHASE00,
               'CPHASE01': CPHASE01,
               'CPHASE10': CPHASE10,
               'CPHASE': CPHASE,
               'SWAP': SWAP,
               'CSWAP': CSWAP,
               'ISWAP': ISWAP,
               'PSWAP': PSWAP,
               'BARENCO': BARENCO
               }

"""
Utility gates for internal QVM use
"""

P0 = np.array([[1, 0], [0, 0]])

P1 = np.array([[0, 0], [0, 1]])

utility_gates = {'P0': P0, 'P1': P1}
