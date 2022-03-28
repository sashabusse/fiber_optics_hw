import numpy as np
from scipy.special import erfc
import utility


def theory_ber(snr_db, order):
    snr_A = utility.db2A(snr_db)
    return erfc(snr_A * np.sqrt(1.5/(order-1)))/np.log2(order)*2*(1-1/np.sqrt(order))
