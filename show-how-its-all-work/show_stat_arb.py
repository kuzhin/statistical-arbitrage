import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import json


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
def create_dataframe(data, active_1_name, active_2_name):#
    df_combined = pd.DataFrame({
        'Active 1': process_data(data[active_1_name]),
        'Active 2': process_data(data[active_2_name])
    })

    df_combined.dropna(inplace=True)
    return df_combined

# --- Расчёт спреда ---
def calculate_spread(df):
    beta = np.polyfit(df['Active 2'], df['Active 1'], 1)[0]
    spread = df['Active 1'] - beta * df['Active 2']
    return spread

# --- График цен Active 1 и Active 2 ---
def plot_prices(df):
    normalized_df = df.copy()
    for col in normalized_df.columns:
        normalized_df[col] = normalized_df[col] / normalized_df[col].iloc[0] * 100

    plt.figure(figsize=(12, 6))
    plt.plot(normalized_df['Active 1'], label='Active 1 (нормализован)', color='blue')
    plt.plot(normalized_df['Active 2'], label='Active 2 (нормализован)', color='orange')

    plt.title('Динамика нормализованных цен Active 1 и Active 2')
    plt.xlabel('Дата')
    plt.ylabel('Цена (нормализованная)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_prices_dual_axis(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Левая ось Y - Active 1 ---
    ax1.plot(df.index, df['Active 1'], label='BTC', color='#4A6FA6', linewidth=2)
    ax1.set_ylabel('BTC', color='#4A6FA6', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='#4A6FA6', labelsize=9)

    # --- Правая ось Y - Active 2 ---
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['Active 2'], label='ETH', color='#C47E5A', linewidth=2)
    ax2.set_ylabel('ETH', color='#C47E5A', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='#C47E5A', labelsize=9)

    # --- Оформление ---
    plt.title('Цены BTC и ETH', fontsize=12, pad=10)

    # Скрываем ось X (даты) и делаем её невидимой
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', which='both', length=0)
    ax1.set_xticks([])

    # Сетка и границы
    ax1.grid(True, axis='y', linestyle=':', linewidth=0.8, alpha=0.7)

    # Убираем рамку вокруг графика
    for spine in ax1.spines.values():
        spine.set_visible(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # Делаем сетку только по Y
    ax1.set_axisbelow(True)

    # --- Легенда ---
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=9, frameon=False)

    # --- Текст под графиком ---
    plt.figtext(0.5, 0.01, 'Исследуемый период: 1 неделя', ha='center', fontsize=9, color='#555555')

    # --- Макет ---
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Место для нижней подписи
    plt.show()

def plot_ratio(df):
    ratio = df['Active 1'] / df['Active 2']

    # Настройки стиля
    plt.style.use('default')
    plt.figure(figsize=(12, 4))

    # Основной график
    plt.plot(ratio.values, label='Active 1 / Active 2', color='#4A6FA6', linewidth=2)

    # Горизонтальные уровни
    mean = ratio.mean()
    std = ratio.std()
    plt.axhline(mean, color='#3E4C5E', linestyle='--', linewidth=1.2, label='Среднее')
    plt.axhline(mean + std, color='#B0B7C6', linestyle=':', linewidth=1, label='+1σ')
    plt.axhline(mean - std, color='#B0B7C6', linestyle=':', linewidth=1, label='-1σ')

    # Оформление осей
    plt.ylabel('Отношение цен', fontsize=10, labelpad=10)
    plt.xlabel('')
    plt.xticks([])
    plt.yticks(fontsize=9)

    # Сетка и границы
    plt.grid(True, axis='y', linestyle=':', linewidth=0.8, alpha=0.7)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_visible(False)

    # Заголовок и легенда
    plt.title('Отношение Active 1 / Active 2', fontsize=12, pad=10)
    plt.legend(loc='upper right', fontsize=9, frameon=False)

    # Текст под графиком
    plt.figtext(0.5, 0.01, 'Исследуемый период: 1 неделя', ha='center', fontsize=9, color='#555555')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Делаем место для нижней подписи
    plt.show()

# --- График с сигналами входа/выхода ---
def plot_signals(df):
    spread = calculate_spread(df)
    mean = spread.mean()
    std = spread.std()

    plt.figure(figsize=(14, 6))

    # График спреда
    plt.plot(spread, label='Динамика цены', color='purple', alpha=0.7)
    plt.axhspan(mean - 1.5 * std, mean + 1.5 * std, color='gray', alpha=0.15, label='±1.5σ зона')
    plt.axhline(mean, color='black', linestyle='--', label='Среднее')
    plt.axhline(mean + 1.5 * std, color='red', linestyle=':', label='+1.5σ')
    plt.axhline(mean - 1.5 * std, color='green', linestyle=':', label='-1.5σ')

    # Вычисление сигналов
    signals = pd.DataFrame(index=spread.index)
    signals['spread'] = spread
    signals['upper'] = mean + 1.5 * std
    signals['lower'] = mean - 1.5 * std
    signals['position'] = np.nan
    signals.loc[signals['spread'] > signals['upper'], 'position'] = -1
    signals.loc[signals['spread'] < signals['lower'], 'position'] = 1
    signals['position'] = signals['position'].ffill()

    # Выставление сигналов к покупке/продаже
    buy_signals = signals[(signals['position'] == 1) & (signals['position'].shift(1) != 1)]
    sell_signals = signals[(signals['position'] == -1) & (signals['position'].shift(1) != -1)]

    # Отображаем только легенду, без точек на графике
    plt.plot([], [], '^', color='green', markersize=10, label='Покупка')
    plt.plot([], [], 'v', color='red', markersize=10, label='Продажа')

    plt.title('Спред с уровнями и сигналами')
    plt.xlabel('Дата')
    plt.ylabel('Спред')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === Основной запуск ===
if __name__ == "__main__":
    data = load_data()
    active_1 = "ADAUSDT"#ADAUSDT
    active_2 = "ETHUSDT"#XEMUSDT

    df = create_dataframe(data, active_1, active_2)#
    # plot_prices(df)
    plot_signals(df)
    # plot_prices_dual_axis(df)
    # plot_ratio(df)