from dataclasses import dataclass, field

import pandas as pd

from .utils import get_clean_factor_and_forward_returns


@dataclass
class CrossSectionalAnalytics:
    """
    Factor backtest output analytics
    """
    information_coefficient: pd.DataFrame
    quantile_returns: pd.DataFrame


@dataclass
class FactorBacktest:
    """
    Factor Backtest

    Args:
        forward_returns(pd.DataFrame): Forward returns indexed by date and security
        factor_exposures(pd.DataFrame): Factor exposures indexed by data and security
        groups(pd.DataFrame, None): optinal grouping variable indexed by of date and security
    """
    forward_returns: pd.DataFrame
    factor_exposures: pd.DataFrame
    groups: pd.DataFrame
    quantiles: int = 5
    _clean_data: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        self._clean_data = get_clean_factor_and_forward_returns(
            factor=self.factor_exposures,
            quantiles=self.quantiles,
            groupby=self.groups,
            prices=None
        )

    def __call__(self, *args, **kwargs):
        return
