"""
Black-Litterman Asset Allocation.
This is a work in progress
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .math_utils import check_positive_semi_definite


@dataclass
class BlackLitterman:
    """
    Black-Litterman Asset Allocation Model class
    """
    asset_covariance: pd.DataFrame
    views_uncertainty: pd.DataFrame
    implied_excess_returns: pd.Series
    views: pd.DataFrame
    views_link_matrix: pd.DataFrame
    tau: float = 1
    risk_aversion = 2
    name: str = 'BL - Default Model'

    def __post_init__(self):
        # check of all the data and dimensions are correct for estimate black-litterman
        check_positive_semi_definite(self.asset_covariance, 'Asset Covariance')
        check_positive_semi_definite(self.views_uncertainty, 'Views Uncertainity')

    def __call__(self, *args, **kwargs) -> pd.Series:
        """
        estimate black-litterman equilibrium excess returns
        :return:
        """
        tau = self.tau
        assets_idx = self.implied_excess_returns.index
        p = self.views_link_matrix.to_numpy()
        omega = self.views_uncertainty.to_numpy()
        asset_cov = self.asset_covariance.to_numpy()

        term_1 = np.linalg.inv(tau * asset_cov) + np.linalg.multi_dot([p.T, np.linalg.inv(omega), p])
        result = pd.Series(term_1, index=assets_idx)
        return result

    def __repr__(self):
        return f'BlackLitterman - {self.name}'
