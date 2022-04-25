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


def nmse(ref, est):
    ref = np.array(ref, dtype=complex)
    est = np.array(est, dtype=complex)
    e = ref-est
    ee = np.mean(e * e.conj())
    norm = np.mean(ref * ref.conj())
    return np.abs(ee/norm)


def nmse_db(ref, est):
    return P2db(nmse(ref, est))
