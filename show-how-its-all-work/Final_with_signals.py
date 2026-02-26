import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS
from statsmodels.tsa.stattools import adfuller
import json

# --- Загрузка данных ---
def load_data(file_path='D:/Git/Home-project/1_price_list.json'):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def process_pair(prices):
    df = pd.DataFrame(prices, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df['close'] = df['close'].astype(float)
    df.set_index('timestamp', inplace=True)
    return df['close']

def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')


from statsmodels.api import OLS
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd


def plot_cointegration_spread1(data_dict, pair1, pair2, hedge_ratio=None, sigma_level=1.5, window=None):
    """
    Анализ коинтеграции между двумя парами и построение графика спреда.

    Параметры:
        data_dict (dict): Словарь с данными по парам, например: {'ADAUSDT': [...], 'XEMUSDT': [...]}
        pair1 (str): Название первой пары (ключ из data_dict)
        pair2 (str): Название второй пары (ключ из data_dict)
        hedge_ratio (float or None): Коэффициент хеджирования (если None — оценивается через регрессию)
        sigma_level (float): Уровень сигмы для границ (например, 1.5)
        window (int or None): Размер окна для скользящих средних (если None — используется фиксированное среднее)
    """
    # Загрузка и объединение данных
    df1 = process_pair(data_dict[pair1]).rename(pair1)
    df2 = process_pair(data_dict[pair2]).rename(pair2)
    df = pd.concat([df1, df2], axis=1).dropna()

    # Расчёт коэффициента хеджирования
    if hedge_ratio is None:
        model = OLS(df[pair1], df[pair2]).fit()
        hedge_ratio = model.params[0]

    # Расчёт спреда
    df['spread'] = df[pair1] - hedge_ratio * df[pair2]

    # Статистика ADF
    print(f"\nADF тест для спреда между {pair1} и {pair2}:")
    adf_test(df['spread'])

    # Расчёт среднего и отклонений
    if window is not None:
        mean_spread = df['spread'].rolling(window=window).mean()
        std_spread = df['spread'].rolling(window=window).std()
    else:
        mean_spread = df['spread'].mean()
        std_spread = df['spread'].std()

    df['mean'] = mean_spread
    df['upper_bound'] = mean_spread + sigma_level * std_spread
    df['lower_bound'] = mean_spread - sigma_level * std_spread

    # Визуализация
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['spread'], label='Spread', color='blue')
    plt.plot(df.index, df['mean'], linestyle='--', color='black', label='Mean')

    # Заливка области ±sigma_level*sigma
    plt.fill_between(df.index, df['lower_bound'], df['upper_bound'],
                     color='gray', alpha=0.3, label=f'±{sigma_level}σ')

    plt.title(f'Spread between {pair1} and {pair2} with ±{sigma_level}σ bounds')
    plt.xlabel('Time')
    plt.ylabel('Spread')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def plot_cointegration_spread2(data_dict, pair1, pair2, hedge_ratio=None, sigma_levels=[1.1, 1.5], window=None):
    """
    Анализ коинтеграции между двумя парами и построение графика спреда с несколькими уровнями сигм.

    Параметры:
        data_dict (dict): Словарь с данными по парам, например: {'ADAUSDT': [...], 'XEMUSDT': [...]}
        pair1 (str): Название первой пары
        pair2 (str): Название второй пары
        hedge_ratio (float or None): Коэффициент хеджирования (если None — оценивается через регрессию)
        sigma_levels (list of float): Уровни сигм для отрисовки (например, [1.1, 1.5])
        window (int or None): Размер окна для скользящих средних (если None — используется фиксированное среднее)
    """
    # --- 1. Подготовка данных ---
    df1 = process_pair(data_dict[pair1]).rename(pair1)
    df2 = process_pair(data_dict[pair2]).rename(pair2)
    df = pd.concat([df1, df2], axis=1).dropna()

    # --- 2. Расчёт коэффициента хеджирования ---
    if hedge_ratio is None:
        model = OLS(df[pair1], df[pair2]).fit()
        hedge_ratio = model.params[0]

    # --- 3. Расчёт спреда ---
    df['spread'] = df[pair1] - hedge_ratio * df[pair2]

    # --- 4. ADF тест ---
    print(f"\nADF тест для спреда между {pair1} и {pair2}:")
    adf_test(df['spread'])

    # --- 5. Расчёт среднего и сигм ---
    if window is not None:
        mean_spread = df['spread'].rolling(window=window).mean()
        std_spread = df['spread'].rolling(window=window).std()
    else:
        mean_spread = df['spread'].mean()
        std_spread = df['spread'].std()

    df['mean'] = mean_spread

    # Добавляем уровни сигм
    for sigma in sigma_levels:
        df[f'upper_{sigma}'] = mean_spread + sigma * std_spread
        df[f'lower_{sigma}'] = mean_spread - sigma * std_spread

    # --- 6. Визуализация ---
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['spread'], label='Spread', color='blue')
    plt.plot(df.index, df['mean'], linestyle='--', color='black', label='Mean')

    # Основные границы ±1.5σ с заливкой
    plt.fill_between(df.index, df['lower_1.5'], df['upper_1.5'],
                     color='gray', alpha=0.2, label=r'$\pm1.5\sigma$')

    # Линии ±1.1σ и ±1.5σ
    plt.plot(df.index, df['upper_1.1'], linestyle='--', color='orange', label=r'$+1.1\sigma$')
    plt.plot(df.index, df['lower_1.1'], linestyle='--', color='orange', label=r'$-1.1\sigma$')
    plt.plot(df.index, df['upper_1.5'], linestyle='--', color='red', label=r'$+1.5\sigma$', alpha=0.8)
    plt.plot(df.index, df['lower_1.5'], linestyle='--', color='red', label=r'$-1.5\sigma$', alpha=0.8)

    plt.title(f'Spread between {pair1} and {pair2}')
    plt.xlabel('Time')
    plt.ylabel('Spread')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Сигнал появляется в следующей свече после пересечения
def plot_cointegration_spread_true(data_dict, pair1, pair2, hedge_ratio=None, sigma_level=1.5, window=None):
    """
    Анализ коинтеграции между двумя парами и построение графика спреда с индикаторами сигналов.

    Параметры:
        data_dict (dict): Словарь с данными по парам, например: {'ADAUSDT': [...], 'XEMUSDT': [...]}
        pair1 (str): Название первой пары
        pair2 (str): Название второй пары
        hedge_ratio (float or None): Коэффициент хеджирования (если None — оценивается через регрессию)
        sigma_level (float): Уровень сигмы для границ (например, 1.5)
        window (int or None): Размер окна для скользящих средних (если None — используется фиксированное среднее)
    """
    # --- 1. Подготовка данных ---
    df1 = process_pair(data_dict[pair1]).rename(pair1)
    df2 = process_pair(data_dict[pair2]).rename(pair2)
    df = pd.concat([df1, df2], axis=1).dropna()

    # --- 2. Расчёт коэффициента хеджирования ---
    if hedge_ratio is None:
        model = OLS(df[pair1], df[pair2]).fit()
        hedge_ratio = model.params[0]

    # --- 3. Расчёт спреда ---
    df['spread'] = df[pair1] - hedge_ratio * df[pair2]

    # --- 4. ADF тест ---
    print(f"\nADF тест для спреда между {pair1} и {pair2}:")
    adf_test(df['spread'])

    # --- 5. Расчёт среднего и сигм ---
    if window is not None:
        mean_spread = df['spread'].rolling(window=window).mean()
        std_spread = df['spread'].rolling(window=window).std()
    else:
        mean_spread = df['spread'].mean()
        std_spread = df['spread'].std()

    df['mean'] = mean_spread
    df['upper'] = mean_spread + sigma_level * std_spread
    df['lower'] = mean_spread - sigma_level * std_spread

    # --- 6. Генерация сигналов ТОЛЬКО при пересечении границ ---
    df['signal'] = 0  # 0 — нет сигнала

    # Пересечение снизу вверх нижней границы -> покупка
    df['buy_signal'] = ((df['spread'].shift(1) < df['lower'].shift(1)) &
                        (df['spread'] >= df['lower']))

    # Пересечение сверху вниз верхней границы -> продажа
    df['sell_signal'] = ((df['spread'].shift(1) > df['upper'].shift(1)) &
                         (df['spread'] <= df['upper']))

    buy_signals = df[df['buy_signal']].index
    sell_signals = df[df['sell_signal']].index

    # --- 7. Визуализация ---
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['spread'], label='Spread', color='blue')
    plt.plot(df.index, df['mean'], linestyle='--', color='black', label='Mean')

    # Одна общая подпись для двух пунктирных линий
    plt.plot(df.index, df['upper'], linestyle='--', color='red', alpha=0.8, label=r'$\pm1.5\sigma$')
    plt.plot(df.index, df['lower'], linestyle='--', color='red', alpha=0.8)

    # Заливка области ±1.5σ
    plt.fill_between(df.index, df['lower'], df['upper'], color='gray', alpha=0.2)

    # Индикаторы сигналов (только Buy и Sell)
    plt.scatter(buy_signals, df.loc[buy_signals, 'spread'], marker='^', color='green', s=100,
                label='Buy Signal', edgecolor='black')
    plt.scatter(sell_signals, df.loc[sell_signals, 'spread'], marker='v', color='red', s=100,
                label='Sell Signal', edgecolor='black')

    plt.title(f'Spread between {pair1} and {pair2} with Trading Signals')
    plt.xlabel('Time')
    plt.ylabel('Spread')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cointegration_spread(data_dict, pair1, pair2, hedge_ratio=None, sigma_level=1.5, window=None):
    """
    Анализ коинтеграции между двумя парами и построение графика спреда с индикаторами сигналов ТОЧНО на уровнях.

    Параметры:
        data_dict (dict): Словарь с данными по парам, например: {'ADAUSDT': [...], 'XEMUSDT': [...]}
        pair1 (str): Название первой пары
        pair2 (str): Название второй пары
        hedge_ratio (float or None): Коэффициент хеджирования (если None — оценивается через регрессию)
        sigma_level (float): Уровень сигмы для границ (например, 1.5)
        window (int or None): Размер окна для скользящих средних (если None — используется фиксированное среднее)
    """
    # --- 1. Подготовка данных ---
    df1 = process_pair(data_dict[pair1]).rename(pair1)
    df2 = process_pair(data_dict[pair2]).rename(pair2)
    df = pd.concat([df1, df2], axis=1).dropna()

    # --- 2. Расчёт коэффициента хеджирования ---
    if hedge_ratio is None:
        model = OLS(df[pair1], df[pair2]).fit()
        hedge_ratio = model.params[0]

    # --- 3. Расчёт спреда ---
    df['spread'] = df[pair1] - hedge_ratio * df[pair2]

    # --- 4. ADF тест ---
    print(f"\nADF тест для спреда между {pair1} и {pair2}:")
    adf_test(df['spread'])

    # --- 5. Расчёт среднего и сигм ---
    if window is not None:
        mean_spread = df['spread'].rolling(window=window).mean()
        std_spread = df['spread'].rolling(window=window).std()
    else:
        mean_spread = df['spread'].mean()
        std_spread = df['spread'].std()

    df['mean'] = mean_spread
    df['upper'] = mean_spread + sigma_level * std_spread
    df['lower'] = mean_spread - sigma_level * std_spread

    # --- 6. Поиск точек пересечения границ ---
    buy_points = []
    sell_points = []

    for i in range(1, len(df)):
        prev_spread = df['spread'].iloc[i - 1]
        curr_spread = df['spread'].iloc[i]
        lower_bound = df['lower'].iloc[i]
        upper_bound = df['upper'].iloc[i]

        # Пересечение нижней границы снизу вверх
        if prev_spread < lower_bound and curr_spread >= lower_bound:
            time_prev = df.index[i - 1].timestamp()
            time_curr = df.index[i].timestamp()
            x = np.interp(lower_bound, [prev_spread, curr_spread], [time_prev, time_curr])
            cross_time = pd.to_datetime(x, unit='s')
            buy_points.append((cross_time, lower_bound))

        # Пересечение верхней границы сверху вниз
        elif prev_spread > upper_bound and curr_spread <= upper_bound:
            time_prev = df.index[i - 1].timestamp()
            time_curr = df.index[i].timestamp()
            x = np.interp(upper_bound, [prev_spread, curr_spread], [time_prev, time_curr])
            cross_time = pd.to_datetime(x, unit='s')
            sell_points.append((cross_time, upper_bound))

    # --- 7. Визуализация ---
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['spread'], label='Spread', color='blue', linewidth=1.2)
    plt.plot(df.index, df['mean'], linestyle='--', color='black', linewidth=0.8)

    # Одна общая подпись для двух пунктирных линий
    plt.plot(df.index, df['upper'], linestyle='--', color='red', alpha=0.8, linewidth=0.8, label=r'$\pm1.5\sigma$')
    plt.plot(df.index, df['lower'], linestyle='--', color='red', alpha=0.8, linewidth=0.8)

    # Заливка области ±1.5σ
    plt.fill_between(df.index, df['lower'], df['upper'], color='gray', alpha=0.1)

    # Индикаторы сигналов ТОЧНО на границах
    if buy_points:
        buy_times, buy_values = zip(*buy_points)
        plt.scatter(buy_times, buy_values, marker='^', color='green', s=30,
                    label='Сигнал на покупку', edgecolor='black', linewidth=0.5)

    if sell_points:
        sell_times, sell_values = zip(*sell_points)
        plt.scatter(sell_times, sell_values, marker='v', color='red', s=30,
                    label='Сигнал на продажу', edgecolor='black', linewidth=0.5)

    # Убираем название оси Y
    plt.ylabel('')
    plt.xlabel('')
    plt.title(f'Спред между {pair1} и {pair2}', fontsize=10)

    # Минималистичная легенда в правом верхнем углу
    plt.legend(loc='upper right', fontsize=8, frameon=False)

    # Убираем рамку графика и делаем сетку едва заметной
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.show()
data = load_data()

# plot_cointegration_spread1(data, 'ADAUSDT', 'XEMUSDT',sigma_level=1.5)
# plot_cointegration_spread2(data, 'ADAUSDT', 'XEMUSDT',sigma_levels=[1.1, 1.5])
plot_cointegration_spread(data, 'ADAUSDT', 'XEMUSDT',sigma_level=1.5)
