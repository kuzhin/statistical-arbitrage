from config_strategy_api import session

# Get symbols that are tradeable

# def get_tradeable_symbols():
#
#     # Get available symbols
#     sym_list = []
#     symbols = session.query_symbol()
#     if "ret_msg" in symbols.keys():
#         if symbols["ret_msg"] == "OK":
#             symbols = symbols["result"]
#             for symbol in symbols:
#                 if symbol["quote_currency"] == "USDT" and symbol["status"] == "Trading": # symbol["maker_fee"]) < 0 removed as ByBit changed terms
#                     sym_list.append(symbol)
#
#     # Return ouput
#     return sym_list

def get_tradeable_symbols(include_spot, include_linear, max_tokens):
    """
    Получает список торговых токенов с возможностью ограничения количества

    Параметры:
        include_spot (bool): Включать спотовые токены (по умолчанию False)
        include_linear (bool): Включать линейные фьючерсы (по умолчанию True)
        max_tokens (int|None): Максимальное количество возвращаемых токенов (None - без ограничений)

    Возвращает:
        set: Множество уникальных базовых токенов (например {'BTC', 'ETH', 'SOL'})
    """
    all_coins = set()

    try:
        # 1. Получаем спотовые токены
        if include_spot:
            spot_response = session.get_instruments_info(category="spot")
            if spot_response['retCode'] == 0:
                for item in spot_response['result']['list']:
                    if max_tokens and len(all_coins) >= max_tokens:
                        break
                    all_coins.add(item['baseCoin'])

        # 2. Получаем линейные фьючерсы
        if include_linear and (max_tokens is None or len(all_coins) < max_tokens):
            cursor = None
            while True:
                params = {
                    "category": "linear",
                    "limit": 1000
                }
                if cursor:
                    params["cursor"] = cursor

                linear_response = session.get_instruments_info(**params)
                if linear_response['retCode'] != 0:
                    break

                for item in linear_response['result']['list']:
                    if max_tokens and len(all_coins) >= max_tokens:
                        break
                    all_coins.add(item['baseCoin'])

                if max_tokens and len(all_coins) >= max_tokens:
                    break

                cursor = linear_response['result'].get('nextPageCursor')
                if not cursor:
                    break

    except Exception as e:
        print(f"Ошибка при запросе символов: {e}")

    return all_coins