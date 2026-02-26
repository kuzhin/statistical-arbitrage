from analysis.cointegration import run_cointegration_analysis # 19.05.2025


if __name__ == "__main__":
    # Запуск анализа с параметрами:
    # - timeframe: '5m' (5 минут)
    # - since_days: 30 (последние 30 дней)
    # TODO: force_update = False Сделать флаг нормально
    results = run_cointegration_analysis(timeframe='5m', since_days=10)
    print(results)