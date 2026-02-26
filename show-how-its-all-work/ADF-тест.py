import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import json

# --- Загрузка данных ---
def load_data(file_path='price_list.json'):
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

# --- Расчёт спреда ---
def calculate_spread(df):
    beta = np.polyfit(df['Active 2'], df['Active 1'], 1)[0]
    spread = df['Active 1'] - beta * df['Active 2']
    return spread


def adf_test_with_plot(series, title='ADF тест'):
    """
    Проводит ADF-тест и строит график временного ряда с результатами теста

    series: временной ряд (например, цена, логарифмированная цена, дифференцированная цена)
    title: подпись для графика
    """
    result = adfuller(series.dropna())
    adf_stat = result[0]
    p_value = result[1]
    crit_vals = result[4]

    # Определение стационарности
    is_stationary = p_value < 0.05

    # --- Построение графика ---
    plt.figure(figsize=(14, 6))
    plt.plot(series.index, series.values, label=title, color='#4A6FA6')

    # --- Оформление ---
    plt.title(f'График: {title}', fontsize=12)
    plt.xlabel('')
    plt.ylabel('Значение')
    plt.xticks([])
    plt.yticks(fontsize=9)

    plt.grid(True, linestyle=':', alpha=0.7)

    # Перемещаем легенду в правый верхний угол
    plt.legend(loc='upper left', fontsize=9, frameon=False)

    # --- Текстовый блок с результатами ADF ---
    textstr = '\n'.join((
        f'ADF Statistic: {adf_stat:.3f}',
        f'p-value: {p_value:.3f}',
        f'Критические значения:',
        f'  1%: {crit_vals["1%"]:.3f}',
        f'  5%: {crit_vals["5%"]:.3f}',
        f' 10%: {crit_vals["10%"]:.3f}',
        f'{"Стационарен" if is_stationary else "Нестационарен"}'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    plt.figtext(0.15, 0.15, textstr, fontsize=10, bbox=props, ha='right')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Делаем место для текста внизу
    plt.show()

# 1. Загружаем данные из файла
data = load_data()

# 2. Выбираем пару активов
active_1_name = "ADAUSDT"
active_2_name = "XEMUSDT"

# 3. Создаём DataFrame с ценами
df = create_dataframe(data, active_1_name, active_2_name)

# 4. Рассчитываем спред
spread = calculate_spread(df)

# 5. ADF-тест для спреда
adf_test_with_plot(spread, title=f"Спред {active_1_name} / {active_2_name}")