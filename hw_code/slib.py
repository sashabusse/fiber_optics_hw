import numpy as np
from scipy.special import erfc

# %% Modulating QAM symbols from bit stream by Gray code
# Input: bit stream b, QAM order order_qam
# Output: signal x


def modulate_qam_int(x, Nbits):
    if Nbits == 2:
        constel = np.asarray([-1. - 1.j, +1. - 1.j,
                              -1. + 1.j, +1. + 1.j])
    elif Nbits == 4:
        constel = 0
    else:
        print('Signal: modulate_qam_int ==> Unknown modulation')
    return constel[x]


# %% Modulating QAM symbols from bit stream by Gray code
# Input: bit stream b reshaped with order QAM shape
# Output: signal x
def modulate_qam_alt(x):
    if x.shape[0] == 1:
        x = np.squeeze(x)
    if x.shape[0] == 2:
        return modulate_qam_int(x[0, :] + x[1, :]*2, 2)
    elif x.shape[0] == 4:
        return 0

# %% demodulating QAM symbols from bit stream by Gray code


def demodulate_qam_alt(x, modulation_order):
    dec_x = np.zeros((np.log2(modulation_order).astype(int),
                     x.shape[0]), dtype=np.int8)
    if modulation_order == 4:
        dec_x[0, x.real == 1] = 1
        dec_x[1, x.imag == 1] = 1
    elif modulation_order == 16:
        dec_x = 0
    return dec_x

# %% Hard slicer
# Input: signal X, modulation order
# Output: signal x


def hard_slicer_alt(x, modulation_order):
    if modulation_order == 4:
        xs = np.sign(x.real) + 1j * np.sign(x.imag)
    elif modulation_order == 16:
        xs = 0
    return xs


# Q-factor
def q(x):
    return erfc(np.sqrt(x))

# Theoretical BER for MQAM
# Input: EbN0, modulation order
# Output: BER estimation


def theory_BER(EbN0, modulation_order):
    if modulation_order == 4:
        return q(EbN0) / 2
    elif modulation_order == 16:
        return q(EbN0 / 32) / 2
