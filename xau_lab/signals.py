"""Signal-domain validation with bounded temporary storage."""
import numpy as np


def valid_signals(values):
    for start in range(0, len(values), 4096):
        if not np.isin(values[start:start + 4096], (-1, 0, 1)).all():
            return False
    return True
