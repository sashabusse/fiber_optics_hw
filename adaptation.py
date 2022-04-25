import numpy as np


def LS_adapt(M, tx_iq, rx_iq, tx_bits, rcond=0, reg=0):
    V_range = np.arange(2*M, rx_iq.shape[0]-2*M)
    V = np.zeros((2, V_range.size, 2 * (2*M+1)**2), dtype=complex)

    d = tx_iq[V_range, :] - rx_iq[V_range, :]
    for m in range(-M, M+1):
        for n in range(-M, M+1):
            m_ind = m + M
            n_ind = n + M
            V[0, :, (2*M+1)*m_ind + n_ind] = rx_iq[V_range+m, 0] * \
                rx_iq[V_range+n, 0]*rx_iq[V_range+m+n, 0].conj()
            V[0, :, (2*M+1)**2 + (2*M+1)*m_ind + n_ind] = rx_iq[V_range +
                                                                m, 0]*rx_iq[V_range+n, 1]*rx_iq[V_range+m+n, 1].conj()

            V[1, :, (2*M+1)*m_ind + n_ind] = rx_iq[V_range+m, 1] * \
                rx_iq[V_range+n, 0]*rx_iq[V_range+m+n, 0].conj()
            V[1, :, (2*M+1)**2 + (2*M+1)*m_ind + n_ind] = rx_iq[V_range +
                                                                m, 1]*rx_iq[V_range+n, 1]*rx_iq[V_range+m+n, 1].conj()

    c = np.zeros((2, 2 * (2*M+1)**2), dtype=complex)
    c[0] = np.linalg.pinv(V[0].T.conj()@V[0] + reg * np.eye(V.shape[2]),
                          rcond=rcond)@(V[0].T.conj()@d[:, 0])
    c[1] = np.linalg.pinv(V[1].T.conj()@V[1] + reg * np.eye(V.shape[2]),
                          rcond=rcond)@(V[1].T.conj()@d[:, 1])

    d_est = np.zeros_like(d)
    d_est[:, 0] = V[0]@c[0]
    d_est[:, 1] = V[1]@c[1]

    rx_iq_out = rx_iq[V_range, :] + d_est
    tx_iq_out = tx_iq[V_range, :]
    tx_bits_out = tx_bits[V_range[0]*4:(V_range[-1] + 1)*4, :]

    return tx_iq_out, tx_bits_out, rx_iq_out, V, c
