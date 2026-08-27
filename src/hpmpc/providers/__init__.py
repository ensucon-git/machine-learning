"""Direct data sources, so the controller does not depend on Home Assistant
integrations being installed and healthy.

Both providers are free, keyless and Swedish-relevant:

* :mod:`hpmpc.providers.smhi` - SMHI's open meteorological forecast, the same
  model that feeds the national forecast. Point forecasts, ~10 days ahead.
* :mod:`hpmpc.providers.elpris` - day-ahead spot prices per bidding area from
  elprisetjustnu.se, which republishes Nord Pool's data.

Home Assistant remains the fallback for both, and remains the only route for
the house's own sensors.
"""

from .elpris import PriceUnavailable, fetch_prices
from .geocode import geocode
from .smhi import fetch_forecast

__all__ = ["fetch_forecast", "fetch_prices", "geocode", "PriceUnavailable"]
