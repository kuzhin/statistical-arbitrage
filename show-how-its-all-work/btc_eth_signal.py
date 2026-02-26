import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
import time
from strategy.config_strategy_api import *
import json
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.ticker import FuncFormatter


def plot_price(df, price_col):
    """
    Визуализирует динамику цены без уровней и сигналов.

    Параметры:
    - df: DataFrame с данными (должен содержать указанную колонку)
    - price_col: имя колонки с ценой (например, 'Active 1')
    """

    if price_col not in df.columns:
        raise KeyError(f"Колонка '{price_col}' отсутствует в данных. Доступные колонки: {list(df.columns)}")

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df[price_col], label=f'Цена закрытия', color='#4A6FA6', linewidth=2)

    # --- Оформление ---
    plt.title('Динамика цены BTC', fontsize=12, pad=10)
    plt.xlabel('')
    plt.ylabel('Цена', fontsize=10, labelpad=10)

    # Скрываем даты на оси X
    plt.xticks([])
    plt.yticks(fontsize=9)

    # Сетка и границы
    plt.grid(True, axis='y', linestyle=':', linewidth=0.8, alpha=0.7)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_visible(False)

    plt.legend(loc='upper right', fontsize=9, frameon=False)

    # Текст под графиком
    plt.figtext(0.5, 0.08,
                'Исследуемый период: 6 месяцев',
                ha='center', fontsize=12, color='black')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()

def plot_price_with_trend(df, price_col='close'):
    if price_col not in df.columns:
        raise KeyError(f"Колонка '{price_col}' отсутствует в данных.")

    x = np.arange(len(df))
    y = df[price_col].values
    slope, intercept = np.polyfit(x, y, 1)
    trend_line = slope * x + intercept

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, y, label='Цена', color='#4A6FA6', linewidth=2)
    plt.plot(df.index, trend_line, color='red', linestyle='--', linewidth=1.5, label='Линия тренда')

    # --- Оформление ---
    plt.title('Динамика цены BTC/USDT с линией тренда', fontsize=12)
    plt.xlabel('')
    plt.ylabel('Цена (USDT)', fontsize=10)

    plt.xticks([])
    plt.yticks(fontsize=9)
    plt.grid(True, axis='y', linestyle=':', alpha=0.7)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_visible(False)

    plt.legend(loc='upper right', fontsize=9, frameon=False)
    plt.figtext(0.5, 0.08, 'Исследуемый период: 6 месяцев', ha='center', fontsize=9, color='black')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()

# === Функция получения данных с Bybit ===
def get_bybit_kline(symbol="BTCUSDT", interval="60", limit=200, save_to_file=True):
    os.makedirs("data", exist_ok=True)

    end_time = int(datetime.now().timestamp())
    start_time = int((datetime.now() - timedelta(days=180)).timestamp())

    all_klines = []

    print(f"Загрузка данных для {symbol}...")

    while start_time < end_time:
        kline = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            start=start_time * 1000,
            limit=limit
        )

        data = kline.get('result', {}).get('list', [])
        if not data:
            print("Нет данных по запросу.")
            break

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df.set_index('timestamp', inplace=True)
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)

        all_klines.append(df)
        start_time = int(df.index[-1].timestamp()) + 60  # Переход к следующей точке
        time.sleep(0.1)

    full_df = pd.concat(all_klines)
    full_df = full_df[~full_df.index.duplicated(keep='first')]
    print(f"Загружено {len(full_df)} записей.")

    if save_to_file:
        filename_json = f"data/{symbol}_{interval}min.json"
        filename_csv = f"data/{symbol}_{interval}min.csv"

        records = full_df.reset_index().to_dict('records')
        with open(filename_json, 'w') as f:
            json.dump(records, f, indent=2, default=str)
        print(f"Данные сохранены в {filename_json}")

        full_df.to_csv(filename_csv)
        print(f"Данные сохранены в {filename_csv}")

    return full_df

def plot_btc_daily_change(df):
    df = df.copy()
    df['change'] = df['close'].pct_change() * 100  # Процентное изменение

    plt.figure(figsize=(14, 4))
    plt.bar(df.index, df['change'], label='Дневное изменение (%)', color='purple', alpha=0.6)

    # --- Оформление ---
    plt.title('Ежедневное изменение цены BTC (в %)', fontsize=12, pad=10)
    plt.xlabel('')
    plt.ylabel('Изменение (%)')
    plt.xticks([])
    plt.yticks(fontsize=9)

    plt.axhline(0, color='black', linewidth=0.8)
    plt.axhline(y=df['change'].std(), color='red', linestyle=':', linewidth=1, label='+1σ')
    plt.axhline(y=-df['change'].std(), color='green', linestyle=':', linewidth=1, label='-1σ')

    plt.grid(True, axis='y', linestyle=':', linewidth=0.8, alpha=0.7)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_visible(False)

    plt.legend(loc='upper right', fontsize=9, frameon=False)
    plt.figtext(0.5, 0.08,
                'Исследуемый период: 6 месяцев',
                ha='center', fontsize=12, color='black')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()
def plot_btc_volatility_clusters(df):
    df = df.copy()
    df['volatility'] = df['close'].pct_change().rolling(window=30).std() * 100  # 30-дневная волатильность %

    plt.figure(figsize=(14, 4))
    plt.plot(df.index, df['volatility'], label='Волатильность (30 дней)', color='orange', linewidth=2)

    plt.title('Волатильность BTC (процентное изменение)', fontsize=12, pad=10)
    plt.xlabel('')
    plt.ylabel('Волатильность (%)')
    plt.xticks([])
    plt.yticks(fontsize=9)

    plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_visible(False)

    plt.legend(loc='upper right', fontsize=9, frameon=False)
    plt.figtext(0.5, 0.08, 'Исследуемый период: 6 месяцев', ha='center', fontsize=9, color='black')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()


def simulate_gbm(S0, mu, sigma, T, dt=1 / 24):
    """
    S0: начальная цена
    mu: средняя доходность (дрейф)
    sigma: волатильность
    T: количество дней
    dt: шаг времени (например, 1/24 — для часовых данных)
    """
    N = int(T / dt)
    times = np.linspace(0, T, N)
    prices = np.zeros(N)
    prices[0] = S0

    for i in range(1, N):
        dB = np.random.normal(0, np.sqrt(dt))
        prices[i] = prices[i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dB)

    return pd.Series(prices, index=pd.date_range(start='2024-10-01', periods=N, freq='h'))
# --- Подготовка данных ---
def simulate_gbm_paths(S0, mu, sigma, T, dt=1 / 24, n_paths=5):
    """
    Генерация нескольких траекторий GBM

    S0: начальная цена
    mu: дрейф (можно установить 0)
    sigma: волатильность
    T: количество дней
    dt: шаг времени
    n_paths: количество траекторий
    """
    N = int(T / dt)
    times = pd.date_range(start='2024-10-01', periods=N, freq='h')
    paths = np.zeros((N, n_paths))
    paths[0] = S0

    for i in range(1, N):
        dB = np.random.normal(0, np.sqrt(dt), size=n_paths)
        paths[i] = paths[i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dB)

    return pd.DataFrame(paths, index=times)


def plot_real_vs_gbm_log_multi(df_btc, mu='rolling', sigma='real', n_paths=5):
    df = df_btc.copy()

    # Оценка параметров
    returns = df['close'].pct_change().dropna()

    if mu == 'rolling':
        mu_val = returns.rolling(window=72).mean().fillna(0.001).mean() * 24 * 365  # годовая
    elif mu == 'zero':
        mu_val = 0.0
    else:
        mu_val = returns.mean() * 24 * 365

    if sigma == 'real':
        sigma_val = returns.std() * np.sqrt(24 * 365)
    else:
        sigma_val = 0.8  # примерное значение

    S0 = df['close'].iloc[0]

    print(f"Параметры GBM:\nS0={S0:.2f}, μ={mu_val:.4f}, σ={sigma_val:.4f}")

    # Генерируем несколько траекторий GBM
    simulated_paths = simulate_gbm_paths(S0=S0, mu=mu_val, sigma=sigma_val, T=len(df), n_paths=n_paths)

    # --- Логарифмируем ---
    df['log_close'] = np.log(df['close'])
    simulated_log = np.log(simulated_paths)

    # --- Визуализация ---
    fig, ax = plt.subplots(figsize=(14, 6))

    # Реальный логарифмированный ряд
    ax.plot(df.index, df['log_close'], label='Реальная цена BTC (ln)', color='#4A6FA6', linewidth=2)

    # Множество смоделированных траекторий
    for col in simulated_log.columns:
        ax.plot(simulated_log.index, simulated_log[col], color='orange', linestyle='--', alpha=0.5)

    # --- Оформление ---
    ax.set_title('Сравнение логарифмированной цены BTC и траекторий GBM', fontsize=12)
    ax.set_ylabel('ln(Цена)')
    ax.set_xlabel('')
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)

    # Убираем ось X
    ax.tick_params(axis='x', which='both', length=0)
    ax.set_xticks([])

    # Убираем рамку
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Легенда
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='#4A6FA6', lw=2),
        Line2D([0], [0], color='orange', lw=1, linestyle='--')
    ]
    ax.legend(custom_lines, ['Реальная цена BTC (ln)', 'Модель GBM'], loc='upper left', fontsize=9, frameon=False)

    # Подпись внизу
    plt.figtext(0.5, 0.08,
                'Исследуемый период: 6 месяцев',
                ha='center', fontsize=9, color='black')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()

# === Основной запуск ===
if __name__ == "__main__":
    # Загружаем готовые данные из файла
    file_path = "data/BTCUSDT_240min.csv"
    btc_data = pd.read_csv(file_path, index_col='timestamp', parse_dates=['timestamp'])
    btc_data.sort_index(inplace=True)
    # plot_btc_volatility_clusters(btc_data)

    # plot_btc_daily_change(btc_data)

    # Построение графика
    # plot_price(btc_data, price_col='close')  # если в файле есть 'close'

    plot_real_vs_gbm_log_multi(btc_data, mu='rolling', sigma='real', n_paths=10)
    # 1. График цены с трендом
    # plot_price_with_trend(btc_data, price_col='close')