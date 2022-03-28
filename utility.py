import numpy as np


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
