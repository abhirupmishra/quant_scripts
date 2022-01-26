import numpy as np
import pandas as pd


def check_positive_semi_definite(matrix: pd.DataFrame, name: str):
    """
    check if matrix is positive semi definite
    :param matrix:
    :param name:
    :return:
    """
    if any(np.linalg.eigvals(matrix) < 0):
        raise ValueError(f'{name} matrix isn\'nt positive semi-definite')
