from pathlib import Path
import numpy as np


def loadCS(filepath: Path, key: str, lmax: int):
    """

    :param filepath: path of SH file
    :param key: '' if there is not any key.
    :param lmax: max degree and order.
    :param lmcs_in_queue: iter, Number of columns where degree l, order m, coefficient Clm, and Slm are located.
    :return: 2d tuple, whose elements are clm and slm in form of 2d array.
    """

    l_queue = 2
    m_queue = 3
    c_queue = 4
    s_queue = 5

    mat_shape = (lmax + 1, lmax + 1)
    Clm, Slm = np.zeros(mat_shape), np.zeros(mat_shape)

    with open(filepath) as f:
        txt_list = f.readlines()
        for i in range(len(txt_list)):
            if txt_list[i].startswith(key):
                this_line = txt_list[i].split()

                l = int(this_line[l_queue - 1])
                if l > lmax:
                    continue

                m = int(this_line[m_queue - 1])

                Clm[l, m] = float(this_line[c_queue - 1])
                Slm[l, m] = float(this_line[s_queue - 1])

    return np.array(Clm), np.array(Slm)
