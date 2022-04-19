import numpy as np
import scipy
from scipy import signal, optimize, io
import pandas as pd

import warnings

from matplotlib import pyplot as plt
import seaborn as sns

import utility
from utility import A2db, P2db, db2A, db2P

from mapping import qam16_ser_labeled, qam16_demap2bits_labeled

mat_data = io.loadmat('pbm_test.mat')
skip_data_sym = 100
tx_bits = mat_data['srcPermBitData'][4*skip_data_sym:-4*skip_data_sym, :]
tx_iq = mat_data['srcSymData'][skip_data_sym:-skip_data_sym, :]
rx_iq = mat_data['eqSymOutData'][skip_data_sym:-skip_data_sym, :]

print("NMSE: {} dB".format(utility.nmse_db(rx_iq, tx_iq)))
ser0 = qam16_ser_labeled(rx_iq[:, 0], tx_iq[:, 0])
ser1 = qam16_ser_labeled(rx_iq[:, 1], tx_iq[:, 1])

rx_bits = np.zeros_like(tx_bits)
rx_bits[:, 0] = qam16_demap2bits_labeled(
    rx_iq[:, 0], tx_iq[:, 0], tx_bits[:, 0])
rx_bits[:, 1] = qam16_demap2bits_labeled(
    rx_iq[:, 1], tx_iq[:, 1], tx_bits[:, 1])
ber0 = np.mean(rx_bits[:, 0] ^ tx_bits[:, 0])
ber1 = np.mean(rx_bits[:, 1] ^ tx_bits[:, 1])

print("SER0: {}".format(ser0))
print("SER1: {}".format(ser1))
print("SER_AVG: {}".format(np.mean([ser0, ser1])))
print("BER0: {}".format(ber0))
print("BER1: {}".format(ber1))
print("BER_AVG: {}".format(np.mean([ber0, ber1])))
