"""
Sets up the appropriate QVM, and interfacing.
"""
from referenceqvm.gates import gate_matrix
from referenceqvm.qvm_wavefunction import QVM_Wavefunction
from referenceqvm.qvm_unitary import QVM_Unitary


def SyncConnection(type_trans='wavefunction',
                   gate_set=gate_matrix):
        """
        Initialize a qvm of a particular type. The type corresponds to the
        type of transition the QVM can perform.

        Currently available transitions:
            wavefunction
            unitary
            (density) support to be implemented
    
        'Wavefunction' is set by default. No noise/t1/t2 params are supported
        in this ref-qvm yet. The QVM uses the gate_set in the
        gate_matrix dictionary by default

        :param type_trans: Transition type of the qvm. Either wavefunction or unitary
        :param gate_set: The set of gates that each qubit or pair of qubits can perform
        """

        if type_trans == 'wavefunction':
            qvm = QVM_Wavefunction(gate_set=gate_set)

        elif type_trans == 'unitary':
            qvm = QVM_Unitary(gate_set=gate_set)

        else:
            raise TypeError("{} is not a valid QVM type.".format(type_trans))

        return qvm
