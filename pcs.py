import functools
import warnings
import numpy as np
import pandas as pd
from scipy import optimize


# ideal probability constellation shaping utility
def _qam_C_optim(alpha, C):
    pts_sqr = np.array([2, 10, 10, 18], dtype=float)
    p_pts = np.exp(-alpha*pts_sqr)
    p_pts /= np.sum(p_pts)

    res = -C - np.sum(p_pts*np.log2(p_pts)) + 2
    return res


def qam_optimal_p0(C):
    alpha = optimize.root_scalar(lambda x: _qam_C_optim(x, C),
                                 bracket=[0, 1],
                                 method='brentq').root
    p0 = np.exp(-alpha)
    p1 = np.exp(-9*alpha)
    norm = p0 + p1
    p0 /= norm
    p1 /= norm
    return p0, p1


# ccdm implimentation
def val2arr(val, bits):
    res = np.zeros(bits, dtype=int)
    for i in range(bits):
        res[i] = (val >> i) & 1
    return res


def arr2val(arr):
    return np.sum(arr * 2**np.arange(arr.size))


@functools.cache
def Comb(n, k):
    assert n >= k and n >= 0 and k >= 0, "bad arguments: n={}, k={}".format(
        n, k)
    if n == k:
        return 1
    if k == 0:
        return 1

    return Comb(n-1, k-1) + Comb(n-1, k)


def idx(x, w, y_sz, warnings_on=False):

    if w != np.sum(x):
        if warnings_on:
            warnings.warn("Bad Sequence w != sum(x): {} != {}, ignoring {} MSB ones".format(
                w, np.sum(x), x, np.sum(x)-w),
                DeprecationWarning, stacklevel=2)

        to_ignore = np.sum(x)-w
        for i in range(x.size):
            if x[-i] == 1:
                x[-i] = 0
                to_ignore -= 1
                if to_ignore == 0:
                    break

    res = 0
    x_sum = 0
    for i in range(x.size):
        if x[i] != 0:
            n = x.size-i-1
            k = w-x_sum
            x_sum += x[i]
            if k <= n:
                res += Comb(n, k)
    return val2arr(res, y_sz)


def idx_rev(y, w, x_sz):
    x = np.zeros(x_sz, dtype=int)

    y_val = arr2val(y)
    x_sum = 0
    for i in range(x_sz):
        n = x.size-i-1
        k = w-x_sum
        C = 0 if n < k else Comb(n, k)
        if y_val >= C:
            x_sum += 1
            x[i] = 1
            y_val -= C
    return x


def ccdm_encode(data_bits, one_bits, d_sz, c_sz):
    data_bits = np.array(data_bits, dtype=int)
    if data_bits.size % d_sz != 0:
        warnings.warn('data.size not divisible by d_sz: {}%{}={}, zeros appended'.format(
            data_bits.size, d_sz, data_bits.size % d_sz),
            DeprecationWarning, stacklevel=2)

        data_bits = np.hstack(
            [data_bits, np.zeros(data_bits.size % d_sz, dtype=int)])

    code_bits = np.zeros(data_bits.size//d_sz*c_sz, dtype=int)

    for i in range(data_bits.size//d_sz):
        code_bits[i*c_sz:(i+1)*c_sz] = idx_rev(data_bits[i *
                                                         d_sz:(i+1)*d_sz], one_bits, c_sz)

    return code_bits


def ccdm_decode(code_bits, one_bits, d_sz, c_sz):
    assert code_bits.size % c_sz == 0, 'code_bits.size % c_sz != 0: {} % {} = {}'.format(
        code_bits.size, c_sz, code_bits.size % c_sz)

    data_bits = np.zeros(code_bits.size//c_sz*d_sz, dtype=int)
    for i in range(code_bits.size//c_sz):
        data_bits[i*d_sz:(i+1)*d_sz] = idx(code_bits[i *
                                                     c_sz:(i+1)*c_sz], one_bits, d_sz)

    return data_bits


def ccdm_make_cfg_min_bits(d_sz, c_sz):
    assert Comb(
        c_sz, c_sz//2) >= 2**d_sz, 'c_sz={} is too low to be used'.format(c_sz)

    one_bits = 1
    while Comb(c_sz, one_bits) < 2**d_sz:
        one_bits += 1

    return pd.Series({
        "one_bits": one_bits,
        "d_sz": d_sz,
        "c_sz": c_sz
    })


def ccdm_make_cfg_probability(d_sz, c_sz, p0):
    assert Comb(
        c_sz, c_sz//2) >= 2**d_sz, 'c_sz={} is too low to be used'.format(c_sz)

    one_bits_min = 1
    while Comb(c_sz, one_bits_min) < 2**d_sz:
        one_bits_min += 1

    p1 = 1-p0
    one_bits_prob = np.rint(p1*c_sz)

    if one_bits_min > one_bits_prob:
        warnings.warn('one_bits_min = {} > one_bits_prob = {}, falling to one_bits_min, p0={:.2f} (desired p0={:.2f})'.format(
            one_bits_min, one_bits_prob, 1-one_bits_min/c_sz, p0),
            DeprecationWarning, stacklevel=2)

    return pd.Series({
        "one_bits": max(one_bits_min, one_bits_prob),
        "d_sz": d_sz,
        "c_sz": c_sz
    })
