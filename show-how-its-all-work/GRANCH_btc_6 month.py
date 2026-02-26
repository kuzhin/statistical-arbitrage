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
from arch import arch_model
from arch.univariate import GARCH



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


# --- Функция для оценки и прогноза волатильности с помощью GARCH ---
def run_garch_forecast(df_series, steps=24, title='GARCH Forecast'):
    """
    Обучает модель GARCH(1,1) и делает прогноз на заданное количество шагов вперёд.

    df_series: pd.Series с ценами (например, close)
    steps: количество часов для прогноза
    """

    # --- Доходности в процентах ---
    returns = 100 * df_series.pct_change().dropna()

    # --- Модель GARCH(1,1) ---
    model = arch_model(returns, vol='GARCH', p=1, o=0, q=1)
    results = model.fit(disp='off')

    # --- Прогноз волатильности ---
    forecast = results.forecast(horizon=steps)
    variance_forecast = forecast.variance.iloc[-1].values
    volatility_forecast = np.sqrt(variance_forecast)

    print("\nПрогноз волатильности (std):")
    forecast_index = pd.date_range(df_series.index[-1], periods=steps + 1, freq='H')[1:]
    for idx, vol in zip(forecast_index, volatility_forecast):
        print(f"{idx.strftime('%Y-%m-%d %H:%M')}: {vol:.5f}")

    # --- Визуализация ---
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Левая ось Y: цена актива
    ax1.plot(df_series.index, df_series.values, label='Цена (BTC/USDT)', color='#4A6FA6')
    ax1.set_xlabel('')
    ax1.set_ylabel('Цена (USDT)', color='#4A6FA6')
    ax1.tick_params(axis='y', labelcolor='#4A6FA6')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Правая ось Y: волатильность
    ax2 = ax1.twinx()

    # Историческая волатильность (rolling std)
    historical_volatility = returns.rolling(window=24).std()
    ax2.plot(historical_volatility.index, historical_volatility, label='Историческая волатильность', color='orange', alpha=0.8)

    # Прогнозная волатильность
    ax2.plot(forecast_index, volatility_forecast, label='Прогноз волатильности (GARCH)', color='red', linestyle='--')

    ax2.set_ylabel('Волатильность (%)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Комбинируем легенды
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=9, frameon=False)

    plt.title(f'{title} | GARCH(1,1) и историческая волатильность', fontsize=12)
    plt.xticks([])
    plt.yticks(fontsize=9)

    plt.figtext(0.8, 0.08,
                'Исследуемый период: ~6 месяцев',
                ha='center', fontsize=9, color='black')

    fig.tight_layout()
    plt.show()

    return {
        'model': results,
        'forecast': volatility_forecast
    }


data = load_data()
btc_series = process_data(data)

# Можно взять последние N записей
btc_series = btc_series[-250:]  # около 6 недель при H1

# # Приводим к явному формату H1
# btc_series = btc_series.sort_index()
# btc_series.index = pd.to_datetime(ada_short.index)
# btc_series = btc_series.asfreq('H').interpolate()

# Запуск GRANCH
result = run_garch_forecast(btc_series, title='BTC/USDT')