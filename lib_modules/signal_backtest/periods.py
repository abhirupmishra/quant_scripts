"""
Time Period Frequency
"""
from enum import Enum

APPROX_BDAYS_PER_MONTH = 21
APPROX_BDAYS_PER_YEAR = 252

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
QTRS_PER_YEAR = 4


class Periods(Enum):
    """
    Time periods
    """
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'
    QUARTERLY = 'quarterly'
    YEARLY = 'yearly'


ANNUALIZATION_FACTORS = {
    Periods.DAILY: APPROX_BDAYS_PER_YEAR,
    Periods.WEEKLY: WEEKS_PER_YEAR,
    Periods.MONTHLY: MONTHS_PER_YEAR,
    Periods.QUARTERLY: QTRS_PER_YEAR,
    Periods.YEARLY: 1
}