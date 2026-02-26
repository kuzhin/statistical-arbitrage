import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Загрузка данных ---
def load_data(file_path='D:/Git/Home-project/show-how-its-all-work/data/BTCUSDT_240min.json'):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# --- Обработка данных ---
def process_data(prices):
    df = pd.DataFrame(prices, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['close'] = df['close'].astype(float)
    df.set_index('timestamp', inplace=True)
    return df['close']


def run_arima_forecast(df_series, steps=24, order=(2, 1, 2), title='ARIMA Forecast'):
    """
    Обучает модель ARIMA и делает прогноз на заданное количество шагов вперёд.

    df_series: pd.Series с ценами (например, close)
    order: параметры модели (p,d,q)
    steps: количество часов для прогноза
    """

    def adf_check(series, name):
        result = adfuller(series.dropna())
        print(f"\n{name}:")
        print(f'ADF Statistic: {result[0]:.3f}')
        print(f'p-value: {result[1]:.3f}')

    print("ADF-тест для исходного ряда:")
    adf_check(df_series, "Original")

    diff_series = df_series.diff().dropna()
    adf_check(diff_series, "Differenced")

    # --- Подготовка временного ряда ---
    if not isinstance(df_series.index, pd.DatetimeIndex):
        raise ValueError("Индекс должен быть DatetimeIndex")

    df_series = df_series.sort_index()
    full_range = pd.date_range(start=df_series.index.min(), end=df_series.index.max(), freq='H')
    df_cleaned = df_series.reindex(full_range).interpolate(method='time')

    # --- Обучаем модель ---
    model = SARIMAX(df_cleaned, order=order, trend='c')
    results = model.fit(disp=False)

    # --- Прогноз ---
    forecast = results.get_forecast(steps=steps)
    predicted_mean = forecast.predicted_mean
    pred_ci = forecast.conf_int()

    # --- Визуализация с двумя осями Y ---
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Левая ось Y: цена актива
    ax1.plot(df_cleaned.index, df_cleaned.values, label='Исторические данные', color='#4A6FA6')
    forecast_index = pd.date_range(df_cleaned.index[-1], periods=steps + 1, freq='H')[1:]
    ax1.plot(forecast_index, predicted_mean, label=f'Прогноз ({order})', color='red', linestyle='--')
    ax1.fill_between(forecast_index,
                     pred_ci.iloc[:, 0],
                     pred_ci.iloc[:, 1], color='pink', alpha=0.3)

    ax1.set_xlabel('')
    ax1.set_ylabel('Цена (USDT)', color='#4A6FA6')
    ax1.tick_params(axis='y', labelcolor='#4A6FA6')
    ax1.grid(True, axis='y', linestyle=':', alpha=0.7)

    # --- Правая ось Y: процентное изменение ---
    ax2 = ax1.twinx()

    # Вычисляем pct_change
    returns = df_cleaned.pct_change() * 100  # в процентах
    ax2.plot(returns.index, returns.values, label='Процентное изменение', color='orange', alpha=0.7)

    ax2.set_ylabel('Изменение (%)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # --- Комбинируем легенды ---
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=9, frameon=False)

    # --- Оформление ---
    plt.title(f'{title} | ARIMA {order} и процентное изменение', fontsize=12)
    plt.xticks([])
    plt.yticks(fontsize=9)

    plt.figtext(0.5, 0.08,
                'Исследуемый период: ~6 месяцев',
                ha='center', fontsize=9, color='black')

    fig.tight_layout()
    plt.show()

    # --- Выводим прогноз ---
    print(f"\nПрогноз на следующие {steps} часов:")
    forecast_df = pd.DataFrame({
        'forecast': predicted_mean.values
    }, index=forecast_index)

    for idx, val in zip(forecast_index, predicted_mean):
        print(f"{idx.strftime('%Y-%m-%d %H:%M')}: {val:.5f}")

    return {
        'model': results,
        'forecast': predicted_mean
    }


data = load_data()
ada_series = process_data(data)

# Можно взять последние N записей
ada_short = ada_series[-250:]  # около 6 недель при H1

# Приводим к явному формату H1
ada_short = ada_short.sort_index()
ada_short.index = pd.to_datetime(ada_short.index)
ada_short = ada_short.asfreq('H').interpolate()

# Запуск ARIMA
result = run_arima_forecast(ada_short, order=(2, 1, 2))