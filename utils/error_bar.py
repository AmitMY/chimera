from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from utils.file_system import temp_name


def error_bar(d: Dict[int, List[int]], y_label: str, x_label: str, ):
    x = sorted(d.keys())
    y = list(map(lambda i: np.mean(d[i]), x))
    e = list(map(lambda i: np.std(d[i]), x))

    plt.errorbar(x, y, e)
    plt.ylabel(y_label, fontsize=18)
    plt.xlabel(x_label, fontsize=18)
    plt.gcf().subplots_adjust(left=0.15)

    tmp = temp_name(suffix=".pdf")
    plt.savefig(tmp)
    plt.close()
    return tmp
