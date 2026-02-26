import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns


import json
from statsmodels.tsa.vector_ar.vecm import coint_johansen
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# TODO: загрузить все токены

# --- Загрузка данных ---
def load_data(file_path='D:/Git/Home-project/1_price_list.json'):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# --- Обработка данных ---
def process_data(prices):
    df = pd.DataFrame(prices, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df['close'] = df['close'].astype(float)
    df.set_index('timestamp', inplace=True)
    return df['close']

# --- Создание DataFrame с двумя активами ---
def create_dataframe(data, active_1_name, active_2_name):
    df_combined = pd.DataFrame({
        'Active 1': process_data(data[active_1_name]),
        'Active 2': process_data(data[active_2_name])
    })

    df_combined.dropna(inplace=True)
    return df_combined


def johansen_test(df, det_order=0, lag=1):
    """
    Выполняет тест Йохансена для проверки коинтеграции

    df: DataFrame с колонками цен активов (например, ['BTC', 'ETH', 'SOL'])
    det_order: детерминированная компонента (-1 = нет, 0 = константа, 1 = линейный тренд)
    lag: количество лагов
    """
    results = coint_johansen(df, det_order=det_order, k_ar_diff=lag)

    # --- Выводим Trace Statistic ---
    print("\nТест Йохансена — Trace Statistic:")
    trace_df = pd.DataFrame({
        'Trace Statistic': results.lr1,
        'Critical Value (90%)': results.cvt[:, 0],
        'Critical Value (95%)': results.cvt[:, 1],
        'Critical Value (99%)': results.cvt[:, 2]
    }, index=[f'r <= {i}' for i in range(len(results.lr1))])
    print(trace_df)

    # --- Выводим Eigenvalue Statistic ---
    print("\nEigenvalue Statistic:")
    eigen_df = pd.DataFrame({
        'Eigen Statistic': results.lr2,
        'Critical Value (90%)': results.cvm[:, 0],
        'Critical Value (95%)': results.cvm[:, 1],
        'Critical Value (99%)': results.cvm[:, 2]
    }, index=[f'r = {i}' for i in range(len(results.lr2))])
    print(eigen_df)

    return {
        'trace_stat': trace_df,
        'eigen_stat': eigen_df,
        'model': results
    }

def plot_correlation_matrix(df, title='Матрица корреляции'):
    """
    Строит тепловую карту корреляции между активами

    df: DataFrame с ценами активов (например, close)
    """
    corr = df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# 1. Загружаем данные из файла
data = load_data()

# 2. Выбираем пару активов
active_1_name = "ADAUSDT"
active_2_name = "XEMUSDT"
# 3. Создаём DataFrame с ценами
df = create_dataframe(data, active_1_name, active_2_name)

# Создаём DataFrame с тремя активами
df_multi = pd.DataFrame({
    'ADA': process_data(data["ADAUSDT"]),
    'XEM': process_data(data["XEMUSDT"]),
    'SOL': process_data(data["SOLUSDT"]),
    'XRP': process_data(data["XRPUSDT"]),
    'XLM': process_data(data["XLMUSDT"]),
}).dropna()

# Тест Йохансена
# print("\nРезультат теста Йохансена для ADA, XEM:")
# johansen_test(df_multi, det_order=0, lag=1)
# beta = [0.3, 0.2]
# plot_cointegrated_spread(df_multi, beta)

# plot_correlation_matrix(df_multi)