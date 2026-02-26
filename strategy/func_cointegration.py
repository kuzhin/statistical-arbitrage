from config_strategy_api import z_score_window
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import pandas as pd
import numpy as np
import math


# Calculate Z-Score
def calculate_zscore(spread):
    df = pd.DataFrame(spread)
    mean = df.rolling(center=False, window=z_score_window).mean()
    std = df.rolling(center=False, window=z_score_window).std()
    x = df.rolling(center=False, window=1).mean()
    df["ZSCORE"] = (x - mean) / std
    return df["ZSCORE"].astype(float).values

# Calculate spread
def calculate_spread(series_1, series_2, hedge_ratio):
    spread = pd.Series(series_1) - (pd.Series(series_2) * hedge_ratio)
    return spread

# Calculate co-integration
def calculate_cointegration(series_1, series_2):
    """
    Расчет коинтеграции
    Args:
        series_1: серия для монеты, например, BTC
        series_2: серия для монеты, например, ETH

    Returns:

    """
    coint_flag = 0
    t_value, p_value, critical_value = coint(series_1, series_2)
    # coint_res = coint(series_1, series_2)
    # t_value = coint_res[0]
    # p_value = coint_res[1]
    # пороговые значения t-статистики: 1% (очень строгий); 5% (стандартный); 10% (мягкий)
    # critical_value = coint_res[2][1]
    model = sm.OLS(series_1, series_2).fit()
    hedge_ratio = model.params[0]
    spread = calculate_spread(series_1, series_2, hedge_ratio)
    zero_crossings = len(np.where(np.diff(np.sign(spread)))[0])
    if p_value < 0.5 and t_value < critical_value[2]:
        coint_flag = 1
    return (coint_flag, round(p_value, 2), round(t_value, 2), round(critical_value[2], 2), round(hedge_ratio, 2), zero_crossings)


# Put close prices into a list
def extract_close_prices(prices):
    close_prices = []
    for candle in prices:
        # Цена закрытия находится под индексом 4
        close_price = candle[4]
        # Преобразуем строку в float
        close_prices.append(float(close_price))
    return close_prices

# Calculate cointegrated pairs
def get_cointegrated_pairs(prices):

    # Loop through coins and check for co-integration
    coint_pair_list = []
    included_list = []
    for sym_1 in prices.keys():

        # Check each coin against the first (sym_1)
        for sym_2 in prices.keys():
            if sym_2 != sym_1:

                # Get unique combination id and ensure one off check
                sorted_characters = sorted(sym_1 + sym_2)
                unique = "".join(sorted_characters)
                if unique in included_list:
                    break
                # Get close prices
                series_1 = extract_close_prices(prices[sym_1])
                series_2 = extract_close_prices(prices[sym_2])

                # Проверка, есть много пар, с разными размерами списков
                if len(series_1) != len(series_2):
                    continue
# бред какой-то с цифрами. Слишком маленький p_value
                # Check for cointegration and add cointegrated pair
                coint_flag, p_value, t_value, c_value, hedge_ratio, zero_crossings = calculate_cointegration(series_1, series_2)

                if coint_flag == 1:
                    included_list.append(unique)
                    coint_pair_list.append({
                        "sym_1": sym_1,
                        "sym_2": sym_2,
                        "p_value": p_value,
                        "t_value": t_value,
                        "c_value": c_value,
                        "hedge_ratio": hedge_ratio,
                        "zero_crossings": zero_crossings
                    })

    # Output results
    df_coint = pd.DataFrame(coint_pair_list)
    df_coint = df_coint.sort_values("zero_crossings", ascending=False)
    df_coint.to_csv("2_cointegrated_pairs.csv")
    return df_coint

