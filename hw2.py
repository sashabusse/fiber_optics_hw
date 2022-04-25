import numpy as np
import scipy
from scipy import signal, optimize, io
import pandas as pd

import warnings

from matplotlib import pyplot as plt
import seaborn as sns

import utility
from utility import A2db, P2db, db2A, db2P

import adaptation
from mapping import qam16_ser_labeled, qam16_demap2bits_labeled


mat_data = io.loadmat('pbm_test.mat')
skip_data_sym = 1000
tx_bits = mat_data['srcPermBitData'][4*skip_data_sym:-4*skip_data_sym, :]
tx_iq = mat_data['srcSymData'][skip_data_sym:-
                               skip_data_sym, :].astype(dtype=complex)
rx_iq = mat_data['eqSymOutData'][skip_data_sym:-
                                 skip_data_sym, :].astype(dtype=complex)


print("initial data parameters:")
print("\tNMSE: {} dB".format(utility.nmse_db(tx_iq, rx_iq)))
ser0 = qam16_ser_labeled(rx_iq[:, 0], tx_iq[:, 0])
ser1 = qam16_ser_labeled(rx_iq[:, 1], tx_iq[:, 1])
print("\tSER0: {}".format(ser0))
print("\tSER1: {}".format(ser1))
print("\tSER_AVG: {}".format(np.mean([ser0, ser1])))

rx_bits = np.zeros_like(tx_bits)
rx_bits[:, 0] = qam16_demap2bits_labeled(
    rx_iq[:, 0], tx_iq[:, 0], tx_bits[:, 0])
rx_bits[:, 1] = qam16_demap2bits_labeled(
    rx_iq[:, 1], tx_iq[:, 1], tx_bits[:, 1])
ber0 = np.mean(rx_bits[:, 0] ^ tx_bits[:, 0])
ber1 = np.mean(rx_bits[:, 1] ^ tx_bits[:, 1])
print("\tBER0: {}".format(ber0))
print("\tBER1: {}".format(ber1))
print("\tBER_AVG: {}".format(np.mean([ber0, ber1])))

# M>12 leads to unable allocate memroy for V
# to implement M higher than 12 we should
# implement other adaptation method
# or get less data
# or implement Vh@V as summation of multiple matrix multiplications
M_range = np.arange(0, 3, 2)
NMSE_res = np.zeros((M_range.size, 3))
SER_res = np.zeros((M_range.size, 3))
BER_res = np.zeros((M_range.size, 3))

for i, M in enumerate(M_range):
    tx_iq_out, tx_bits_out, rx_iq_out, V, c = adaptation.LS_adapt(
        M, tx_iq, rx_iq, tx_bits)

    NMSE_res[i, 0] = utility.nmse_db(tx_iq_out[:, 0], rx_iq_out[:, 0])
    NMSE_res[i, 1] = utility.nmse_db(tx_iq_out[:, 1], rx_iq_out[:, 1])
    NMSE_res[i, 2] = utility.nmse_db(tx_iq_out, rx_iq_out)

    SER_res[i, 0] = qam16_ser_labeled(rx_iq_out[:, 0], tx_iq_out[:, 0])
    SER_res[i, 1] = qam16_ser_labeled(rx_iq_out[:, 1], tx_iq_out[:, 1])
    SER_res[i, 2] = np.mean((SER_res[i, 0], SER_res[i, 1]))

    rx_bits = np.zeros_like(tx_bits_out)
    rx_bits[:, 0] = qam16_demap2bits_labeled(
        rx_iq_out[:, 0], tx_iq_out[:, 0], tx_bits_out[:, 0])
    rx_bits[:, 1] = qam16_demap2bits_labeled(
        rx_iq_out[:, 1], tx_iq_out[:, 1], tx_bits_out[:, 1])
    BER_res[i, 0] = np.mean(rx_bits[:, 0] ^ tx_bits_out[:, 0])
    BER_res[i, 1] = np.mean(rx_bits[:, 1] ^ tx_bits_out[:, 1])
    BER_res[i, 2] = np.mean((BER_res[i, 0], BER_res[i, 1]))
    print('done {}/{}'.format(i+1, M_range.size))

fig = plt.figure(figsize=(9, 5))
#plt.plot(M_range, NMSE_res, label=['NMSE_0', 'NMSE_1', 'NMSE_AVG'])
plt.plot(M_range, NMSE_res[:, 2], label='NMSE_AVG')
plt.xlabel('M')
plt.ylabel('NMSE dB')
plt.legend()
plt.grid(True)

fig = plt.figure(figsize=(9, 5))
#plt.semilogy(M_range, SER_res, label=['SER_0', 'SER_1', 'SER_AVG'])
plt.semilogy(M_range, SER_res[:, 2], label='SER_AVG')
plt.xlabel('M')
plt.ylabel('SER')
plt.legend()
plt.grid(True)

fig = plt.figure(figsize=(9, 5))
#plt.semilogy(M_range, BER_res, label=['BER_0', 'BER_1', 'BER_AVG'])
plt.semilogy(M_range, BER_res[:, 2], label='BER_AVG')
plt.xlabel('M')
plt.ylabel('BER')
plt.legend()
plt.grid(True)

plt.show()
