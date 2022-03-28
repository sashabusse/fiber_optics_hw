import numpy as np
import matplotlib.pyplot as mp
import slib

# Main parameters
N = 500000  # Length of signal
Fs = 10e9  # Frequency band in Hz
OSNR_range = np.linspace(2, 10, 17)  # OSNR range for AWGN in dB
M = 4 # Modulation order
# Initialization
constel = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j])
# Signal generation
b = np.random.choice([0, 1], (2, N))
X = slib.modulate_qam_alt(b)
mp.scatter(constel.real, constel.imag, s=100)
mp.xlim([-2, 2])
mp.ylim([-2, 2])
mp.grid(True)
mp.show()
# Channel
ber_theory = []
ber = []
EbN0_db = []
for osnr in OSNR_range:
    # Converting OSNR to SNR
    snr = osnr - 10 * np.log10(Fs / 12.5e9)
    # Converting to EbN0
    EbN0 = 10 ** (snr / 10) / np.log2(M)
    EbN0_db.append(10 * np.log10(EbN0))
    # Theoretical BER
    ber_theory.append(slib.theory_BER(EbN0, M))
    # Generating noise
    noise = (10 ** (-snr / 20)) * (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2) * np.sqrt(np.var(constel))
    # Adding noise
    X_noisy = X + noise
    if osnr == OSNR_range[0]:
        mp.scatter(X_noisy.real, X_noisy.imag, s=0.01)
        mp.grid(True)
        mp.show()
    # Hard slicing
    X_sliced = slib.hard_slicer_alt(X_noisy, M)
    # Democulation
    b_out = slib.demodulate_qam_alt(X_sliced, M)
    # BER calculation
    ber.append(np.mean(np.abs(b_out - b)))
# Plotting BER vs OSNR curve
mp.semilogy(EbN0_db, ber_theory, 'r--', LineWidth=2)
mp.semilogy(EbN0_db, ber, '*', MarkerSize=10)
mp.xlabel('EbN0, dB')
mp.ylabel('BER')
mp.legend(['Theory', 'Simulation'])
mp.grid(True)
mp.show()

mp.semilogy(OSNR_range, ber_theory, 'r--', LineWidth=2)
mp.semilogy(OSNR_range, ber, '*', MarkerSize=10)
mp.xlabel('OSNR, dB')
mp.ylabel('BER')
mp.legend(['Theory', 'Simulation'])
mp.grid(True)
mp.show()

print('Finish!')
