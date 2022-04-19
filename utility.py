import numpy as np
from scipy import signal


def A2db(A):
    return 20*np.log10(A)


def P2db(P):
    return 10*np.log10(P)


def db2A(db):
    return 10**(db/20)


def db2P(db):
    return 10**(db/10)


def osnr2snr_db(osnr_db, fs):
    return osnr_db - 10 * np.log10(fs / 12.5e9)


def snr2osnr_db(snr_db, fs):
    return snr_db + 10 * np.log10(fs / 12.5e9)


def EbN0_db(snr_db, M):
    return snr_db - 10*np.log10(np.log2(M))


def psd_db(s, nfft=2048, window='blackman', fs=1.):
    f, psd = signal.welch(s, fs=fs, window=window,
                          nperseg=nfft, return_onesided=False)
    return np.fft.fftshift(f), np.fft.fftshift(P2db(psd))


def nmse(x, y):
    x = np.array(x, dtype=complex)
    y = np.array(y, dtype=complex)
    e = x-y
    ee = np.mean(e * e.conj())
    xx = np.mean(x * x.conj())
    return np.abs(ee/xx)


def nmse_db(x, y):
    return P2db(nmse(x, y))
