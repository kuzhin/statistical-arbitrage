from func_price_klines import get_price_klines
import json

# Store price histry for all available pairs
def store_price_history(symbols):
    """
    Получает исторические данные цен для списка токенов и сохраняет в JSON

    Args:
        symbols: Множество базовых токенов (в формате: {'BTC', 'ETH', 'SOL', ...})
    """
    # Преобразуем токены в торговые пары (добавляем USDT)
    trading_pairs = [f"{token}USDT" for token in symbols]

    # Get prices and store in DataFrame
    counts = 0
    price_history_dict = {}
    for pair in trading_pairs:
        price_history = get_price_klines(pair)  # Используем готовую функцию get_price_klines

        if len(price_history) > 0:
            price_history_dict[pair] = price_history
            counts += 1
            print(f"{counts} {pair} stored")
        else:
            print(f"{pair} not stored")

    # Output prices to JSON
    if len(price_history_dict) > 0:
        with open("1_price_list.json", "w") as fp:
            json.dump(price_history_dict, fp, indent=4)
        print("Prices saved successfully.")

    # Return output
    return
