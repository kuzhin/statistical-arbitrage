from strategy.func_cointegration import calculate_cointegration
import matplotlib.pyplot as plt
import pandas as pd

z_score_window = 21

def plot_trends_from_csv(filename="3_backtest_file.csv"):
    # 1. Загрузка данных из CSV
    df = pd.read_csv(filename)

    # 2. Проверим, есть ли нужные столбцы
    required_columns = ['Spread', 'ZScore']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("В CSV отсутствуют необходимые столбцы: Spread, ZScore")

    # 3. Получаем символы (предположим, что это первые две колонки)
    sym_1 = df.columns[1]  # Например: '1000000PEIPEIUSDT'
    sym_2 = df.columns[2]  # Например: '1000000MOGUSDT'

    # 4. Извлекаем цены
    prices_1 = df[sym_1].astype(float).tolist()
    prices_2 = df[sym_2].astype(float).tolist()

    # 5. Извлекаем спред и z-score
    spread = df['Spread'].astype(float).tolist()
    zscore = df['ZScore'].astype(float).fillna(0).tolist()  # Заполняем NaN нулями

    # 6. Расчёт нормированных рядов
    df[f"{sym_1}_pct"] = df[sym_1] / prices_1[0]
    df[f"{sym_2}_pct"] = df[sym_2] / prices_2[0]
    series_1 = df[f"{sym_1}_pct"].astype(float).values
    series_2 = df[f"{sym_2}_pct"].astype(float).values

    # 7. Пересчитаем коинтеграцию (опционально, если нужно обновить статистику)
    coint_flag, p_value, t_value, c_value, hedge_ratio, zero_crossing = calculate_cointegration(prices_1, prices_2)

    # 8. Рисуем графики
    fig, axs = plt.subplots(3, figsize=(16, 8))
    fig.suptitle(f"Price and Spread - {sym_1} vs {sym_2}")
    axs[0].plot(series_1, label=sym_1)
    axs[0].plot(series_2, label=sym_2)
    axs[0].legend()

    axs[1].plot(spread, color='green', label='Spread')
    axs[1].legend()

    axs[2].plot(zscore, color='purple', label='Z-Score')
    axs[2].axhline(0, color='black', linestyle='--')
    axs[2].axhline(1, color='red', linestyle='--')
    axs[2].axhline(-1, color='red', linestyle='--')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

plot_trends_from_csv()
