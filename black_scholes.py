"""
Library for Black-Scholes-Merton Option pricing

The computations are incorrect, for now. I'm working on fixing them.
"""
import math
from dataclasses import dataclass

import scipy


@dataclass
class BlackScholesMerton:
    """
    Black-Scholes-Merton pricing class
    """
    spot_price: float
    strike_price: float
    time_to_maturity: float
    risk_free_rate: float
    volatility: float
    dividend_yield: float = 0.
    is_american_option: bool = False

    def __post_init__(self):
        if any([
            self.spot_price,
            self.strike_price,
            self.time_to_maturity,
            self.volatility,
            self.dividend_yield
        ]) < 0:
            raise ValueError('Cannot allow negative values for either of: '
                             'Spot Price, Strike Price, Time to Maturity or Volatility')

    @property
    def call_value(self) -> float:
        """
        property to get european call option price
        :return:
        """
        norm_cdf = scipy.stats.norm.cdf
        d_one = self._compute_d_one()
        d_two = self._compute_d_two()
        discount = self._compute_discount()
        return (self.spot_price * norm_cdf(d_one)) - (self.strike_price * discount * norm_cdf(d_two))

    @property
    def put_value(self):
        """
        european put value
        :return:
        """
        discount = self._compute_discount()
        return self.call_value + (self.strike_price * self.dividend_yield * discount) - self.spot_price

    @property
    def call_delta(self):
        """
        get delta value of the call option
        :return:
        """
        return self._compute_erf_value(self._compute_d_one())

    @property
    def put_delta(self):
        """
        get delta value of the put option
        :return:
        """
        return 1 - self.call_delta

    @property
    def gamma(self):
        norm_cdf = scipy.stats.norm.cdf
        d_one = self._compute_d_one()
        temp_val = norm_cdf(d_one) / (self.volatility * self.spot_price * math.sqrt(self.time_to_maturity))
        discount = self._compute_discount()
        return discount * temp_val

    @property
    def vega(self):
        """
        vega
        :return:
        """
        norm_cdf = scipy.stats.norm.cdf
        d_one = self._compute_d_one()
        discount = self._compute_discount()
        return discount * self.spot_price * math.sqrt(self.time_to_maturity) * norm_cdf(d_one)

    @property
    def theta(self):
        """
        property for theta
        :return:
        """
        norm_cdf = scipy.stats.norm.cdf
        discount = self._compute_discount()
        d_one = self._compute_d_one()
        d_two = self._compute_d_two()
        term_1 = -discount * self.spot_price * norm_cdf(d_one) * self.volatility / (2 * math.sqrt(self.time_to_maturity))
        term_2 = self.risk_free_rate * discount * self.spot_price * norm_cdf(d_one)
        term_3 = self.risk_free_rate
        return term_1 + term_2 + term_3

    @property
    def rho(self):
        """
        property for rho
        :return:
        """
        return 0

    @property
    def implied_volatility(self):
        return

    @staticmethod
    def _compute_erf_value(value: float):
        """
        private function to compute Error function value
        :param value:
        :return:
        """
        return 0.5 * (1 + scipy.special.erf(value/math.sqrt(2)))

    def _compute_discount(self):
        """
        discount value
        :return:
        """
        return math.e**(-self.risk_free_rate * self.time_to_maturity)

    def _compute_d_one(self):
        """
        d-one value
        :return:
        """
        vol_term = 0.5 * (self.volatility**2)
        s_by_x = math.log(self.spot_price/self.strike_price)
        d_one = s_by_x + (self.risk_free_rate - self.dividend_yield + vol_term) * self.time_to_maturity
        return d_one / (self.volatility * math.sqrt(self.time_to_maturity))

    def _compute_d_two(self):
        """
        d-two value
        :return:
        """
        return self._compute_d_one() - (self.volatility * math.sqrt(self.time_to_maturity))
