import numpy as np
from grove.arbitrary_state.arbitrary_state import create_arbitrary_state
from tqdm import tqdm


def test_generate_arbitrary_states(qvm):
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
