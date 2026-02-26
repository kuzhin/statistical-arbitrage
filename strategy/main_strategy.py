import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from func_get_symbols import get_tradeable_symbols
from func_prices_json import store_price_history
from func_cointegration import get_cointegrated_pairs
from func_plot_trends import plot_trends
import pandas as pd
import json


"""STRATEGY CODE"""
if __name__ == "__main__":

    # STEP 1 - Get list of symbols
    print("Getting symbols...")

    pass
    # Уже загружено 516 монет.

    # sym_response = get_tradeable_symbols(include_spot=False, include_linear=True, max_tokens=50)
    #
    # # STEP 2 - Construct and save price history
    # print("Constructing and saving price data to JSON...")
    # if len(sym_response) > 0:
    #     store_price_history(sym_response)
    pass

    # STEP 3 - Find Cointegrated pairs
    print("Calculating co-integration...")
    with open("1_price_list.json") as json_file:
        price_data = json.load(json_file)
        if len(price_data) > 0:
            coint_pairs = get_cointegrated_pairs(price_data)
