"""
@author: Abhirup
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import cvxpy as cp
import numpy as np
import pandas as pd


class OptimizationException(Exception):

    def __init__(self, message: str):
        self.message = message


@dataclass
class RiskModel:
    """
    class to hold risk model data
    """
    asset_covariance: pd.DataFrame

    @classmethod
    def from_matrix(cls, asset_cov: pd.DataFrame) -> RiskModel:
        """
        construct risk model from asset covariance matrix
        :param asset_cov:
        :return:
        """
        return cls(asset_covariance=asset_cov)

    def __post_init__(self):
        # check if the asset covariance matrix is positive definite and full-rank
        pass

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        return self.asset_covariance

    @property
    def assets(self) -> List[str or int]:
        """
        property to get assets of a risk model
        :return:
        """
        return self.asset_covariance.columns.tolist()


def optimize_portfolio(risk_model: RiskModel) -> (float, pd.Series):
    """
    Optimization solve method to build portfolio from covariance matrix
    :param risk_model: RiskModel
    :return:
    """
    # Add variables
    x = cp.Variable((len(risk_model.assets), 1))
    # Budget and weights constraints
    constraints = [
        cp.sum(x) == 1,  # Budget constraint of weights adding to 1
        x <= 1,  # each weight not going beyond 1
        x >= 0  # Long only
    ]
    # risk value
    variance = cp.quad_form(x, risk_model().to_numpy())
    objective = cp.Minimize(variance)
    try:
        prob = cp.Problem(objective, constraints)
        prob.solve()
    except Exception:
        raise OptimizationException('Could not solve - Generic Optimization Error')
    if prob.status != 'optimal':
        raise OptimizationException('Could not find optimal solution')
    weights = pd.DataFrame(x.value, index=risk_model.assets, columns=['weights'])
    return float(prob.objective.value), weights['weights']


def main():
    assets_idx = ['a', 'b', 'c']
    ret = pd.Series(data=[0.0260023, 0.008100891, 0.073774971], index=assets_idx)
    sigma = [[0.017087987, 0.003298885, 0.001224849],
             [0.003298885, 0.005900944, 0.004488271],
             [0.001224849, 0.004488271, 0.063000818]]
    risk_model = RiskModel.from_matrix(asset_cov=pd.DataFrame(data=sigma, index=assets_idx, columns=assets_idx))

    variance, weights = optimize_portfolio(risk_model=risk_model)
    port_ret = float(np.linalg.multi_dot([weights.transpose(), ret]))
    print("Risk =", math.sqrt(variance))
    print("Return =", port_ret)


if __name__ == "__main__":
    main()
