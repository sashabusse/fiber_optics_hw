import numpy as np
import warnings


def qam_order2bits(order):
    res = np.log2(order)
    assert res % 1 == 0, 'bad order value: {}'.format(order)
    return int(res)


def qam_simmetry_bits(order):
    bits = qam_order2bits(order)
    h_bit = bits - 1
    v_bit = bits//2 - 1
    return h_bit, v_bit


def gray_code(bits):
    bits = int(bits)
    res = np.zeros(2**bits, dtype=int)
    for bit_ind in range(0, bits):
        for i in range(0, 2**bits):
            mirror = (i//(2**(bit_ind + 1))) % 2
            val = (i//(2**bit_ind)) % 2
            res[i] |= (val ^ mirror) << bit_ind

    return res

# returns array of complex that maps bit patterns (symbols) to IQ
# order from 4 (QPSK) // BPSK not supported


def qam_sym2iq_map(order):
    # generate gray code mapping
    bits = qam_order2bits(order)
    width = 2**(bits//2)
    g_code = gray_code(bits//2)
    # to map zeros to the center and ones to peripheral
    g_code ^= (1 << bits//2)-1
    # generate bit patterns to iq mapping
    sym2iq_map = np.zeros((order, ), complex)
    for i in range(0, width):  # iterate over rows of constellation
        for j in range(0, width):  # iterate over columns of constellation
            sym2iq_map[(g_code[(i) % g_code.size] << bits//2) +
                       g_code[(j) % g_code.size]] = 2*(i + 1j*j)

    sym2iq_map -= sym2iq_map.mean()
    return sym2iq_map


def qam_vh_bits(order):
    bits = qam_order2bits(order)
    return bits//2-1, bits-1


def constellation_norm(constellation_map):
    return np.sqrt(np.average(np.abs(constellation_map)**2))


def qam_mapping(b, order):
    bits = qam_order2bits(order)
    if b.size % bits != 0:
        warnings.warn('b.size not divisible by {}, zeros appended'.format(bits),
                      DeprecationWarning, stacklevel=2)
        b = np.hstack([b, np.zeros(bits - (b.size % bits), dtype=int)])

    # get chosen constellation map
    sym2iq_map = qam_sym2iq_map(order)

    # generate iq points based on chosen constellation map
    idx = np.zeros(b.size//bits, dtype=int)
    for i in range(bits):
        idx += b[i::bits] << i

    iq = sym2iq_map[idx]
    return iq


def qam_demapping(iq, order):
    bits = qam_order2bits(order)

    # get chosen constellation map
    sym2iq_map = qam_sym2iq_map(order)

    sym2iq_map_mod = (sym2iq_map - sym2iq_map.real.min() * (1 + 1j))/2
    demap_map = np.zeros(order, dtype=int)
    demap_map[((sym2iq_map_mod.real).astype(int) << (bits // 2)) +
              (sym2iq_map_mod.imag).astype(int)] = np.arange(order, dtype=int)
    b_out = np.zeros(bits*iq.size, dtype=int)

    # modify and saturate
    iq_mod = (iq - sym2iq_map.real.min() * (1 + 1j))/2
    iq_mod.real = np.maximum(iq_mod.real, 0)
    iq_mod.imag = np.maximum(iq_mod.imag, 0)
    iq_mod.real = np.minimum(iq_mod.real, 2**(bits//2)-1)
    iq_mod.imag = np.minimum(iq_mod.imag, 2**(bits//2)-1)

    iq_idx = (np.rint(iq_mod.real).astype(int) << (bits//2)) + \
        np.rint(iq_mod.imag).astype(int)
    sym = demap_map[iq_idx]
    for j in range(bits):
        b_out[j::bits] = (sym >> j) & 1

    return b_out


def qam16_demap_labeled(iq, labels):
    labels_idx = ((labels.real+3)//2 + (labels.imag+3)//2 * 4).astype(int)
    labels_avg = np.zeros(16, dtype=complex)
    for i in range(labels_avg.size):
        labels_avg[i] = np.mean(iq[labels_idx == i])

    idx_demaped = np.zeros_like(labels_idx)
    iq_demaped = np.zeros_like(iq)
    for i in range(iq.size):
        idx_demaped[i] = np.argmin(np.abs(labels_avg - iq[i]))
        iq_demaped[i] = ((idx_demaped[i] % 4)*2 - 3) + \
            1j*((idx_demaped[i]//4)*2 - 3)

    return idx_demaped, iq_demaped


def qam16_ser_labeled(iq, labels):
    _, iq_demaped = qam16_demap_labeled(iq, labels)

    ser = np.mean((labels-iq_demaped) != 0)
    return ser


def qam16_demap2bits_labeled(iq, labels_iq, labels_bits):
    labels_idx = (labels_iq.real+3)//2 + (labels_iq.imag+3)//2 * 4
    idx2bits = np.zeros((16, 4), dtype=int)
    for i in range(idx2bits.shape[0]):
        w = np.argwhere(labels_idx == i)[0][0]
        idx2bits[i] = labels_bits[4*w:4*(w+1)]

    idx_demaped, _ = qam16_demap_labeled(iq, labels_iq)

    bits_out = np.zeros(4*iq.size, dtype=int)
    for i in range(idx_demaped.size):
        bits_out[i*4:(i+1)*4] = idx2bits[idx_demaped[i]]

    return bits_out
