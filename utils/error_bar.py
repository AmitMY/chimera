from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from utils.file_system import temp_name


def error_bar(d: Dict[int, List[int]]):
    x = sorted(d.keys())
    y = list(map(lambda i: np.mean(d[i]), x))
    e = list(map(lambda i: np.std(d[i]), x))

    plt.errorbar(x, y, e)

    tmp = temp_name(suffix=".png")
    plt.savefig(tmp)
    return tmp
