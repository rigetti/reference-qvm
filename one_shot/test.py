import pyquil.quil as pq
import pyquil.api as api
from pyquil.gates import *

# open a synchronous connection
qvm = api.SyncConnection()

p = pq.Program()
p.inst(X(0)).measure(0, 0)

print(p)

classical_regs = [0, 1, 2] # A list of which classical registers to return the values of.

qvm.run(p, classical_regs)