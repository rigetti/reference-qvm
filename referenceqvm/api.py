"""
Sets up the appropriate QVM, and interfacing.
"""
from .gates import gate_matrix
from .qvm import QVM, QVM_Unitary


def Connection(type_trans='wavefunction',
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
        """

        if type_trans == 'wavefunction':
            qvm = QVM(gate_set=gate_matrix)

        elif type_trans == 'unitary':
            qvm = QVM_Unitary(gate_set=gate_matrix)

        else:
            raise TypeError("""{} is not a valid QVM type.""".format(type_trans))

        return qvm
