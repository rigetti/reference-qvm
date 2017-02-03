import numpy as np
import cmath

I = np.array([[1.0, 0.0], [0.0, 1.0]])

X = np.array([[0.0, 1.0], [1.0, 0.0]])

Y = np.array([[0.0, 0.0 - 1.0j], [0.0 + 1.0j, 0.0]])

Z = np.array([[1.0, 0.0], [0.0, -1.0]])

H = (1.0/np.sqrt(2.0))*np.array([[1.0, 1.0], [1.0, -1.0]])

S = np.array([[1.0, 0.0], [0.0, 1.0j]])

T = np.array([[1.0, 0.0], [0.0, cmath.exp(-1.0j * np.pi / 4.0)]])

P0 = np.array([[1, 0], [0, 0]])

P1 = np.array([[0, 0], [0, 1]])

UA = np.array([[1, 0, 0, 0],
               [0, 1j, 0, 0],
               [0, 0, 1j, 0],
               [0, 0, 0, 1]])

SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])


def RX(phi):
    return np.array([[np.cos(phi/2.0), -1j*np.sin(phi/2.0)],
                     [-1j*np.sin(phi/2.0), np.cos(phi/2.0)]])


def RY(phi):
    return np.array([[np.cos(phi/2.0), -np.sin(phi/2.0)],
                     [np.sin(phi/2.0), np.cos(phi/2.0)]])


def RZ(phi):
    return np.array([[np.cos(phi/2.0) - 1j*np.sin(phi/2.0), 0],
                     [0, np.cos(phi/2.0) + 1j*np.sin(phi/2.0)]])


def PHASE(phi):
    return np.array([[1.0, 0.0], [0.0, np.exp(1j*phi)]])


def CPHASE(phi):
    return np.diag([1.0, 1.0, 1.0, np.exp(1j*phi)])


def CPHASE00(phi):
    return np.diag([np.exp(1j*phi), 1.0, 1.0, 1.0])


def CPHASE01(phi):
    return np.diag([1.0, np.exp(1j*phi), 1.0, 1.0])


def CPHASE10(phi):
    return np.diag([1.0, 1.0, np.exp(1j*phi), 1.0])


def BARENCO(alpha, phi, theta):
    lower_unitary = np.array([[np.exp(1j*phi)*np.cos(theta), -1j*np.exp(1j*(alpha - phi))*np.sin(theta)],
                              [-1j*np.exp(1j*(alpha + phi))*np.sin(theta), np.exp(1j*alpha)*np.cos(theta)]])

    return np.diag(np.eye(2), lower_unitary)


gate_matrix = {'I': I,
               'X': X,
               'Y': Y,
               'Z': Z,
               'H': H,
               'S': S,
               'T': T,
               'P0': P0,
               'P1': P1,
               'UA': UA,
               'SWAP': SWAP,
               'CNOT': CNOT,
               'RX': RX,
               'RY': RY,
               'RZ': RZ,
               'PHASE': PHASE,
               'CPHASE': CPHASE,
               'CPHASE00': CPHASE00,
               'CPHASE01': CPHASE01,
               'CPHASE10': CPHASE10,
               'BARENCO': BARENCO
               }
