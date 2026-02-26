"""
    interval: 60, "D"
    from: integer from timestamp in seconds
    limit: max size of 200
"""

from config_strategy_api import session
from config_strategy_api import timeframe
from config_strategy_api import kline_limit
import datetime
import time

# Get start times
time_start_date = 0
if timeframe == 60:
    time_start_date = datetime.datetime.now() - datetime.timedelta(hours=kline_limit)
if timeframe == "D":
    time_start_date = datetime.datetime.now() - datetime.timedelta(days=kline_limit)
time_start_seconds = int(time_start_date.timestamp())

# Get historical prices (klines)
# def get_price_klines(symbol):
#
#     # Get prices
#     prices = session.query_mark_price_kline(
#         symbol = symbol,
#         interval = timeframe,
#         limit = kline_limit,
#         from_time = time_start_seconds
#     )
#
#     # Manage API calls
#     time.sleep(0.1)
#
#     # Return output
#     if len(prices["result"]) != kline_limit:
#         return []
#     return prices["result"]


def get_price_klines(symbol, interval=60, limit=200, start_time=None, end_time=None):
    """
    Получает исторические данные цен (K-Lines) через официальную библиотеку Bybit

    Параметры:
        symbol (str): Торговый символ (например "BTCUSDT")
        interval (int): Таймфрейм в минутах (1,3,5,15,30,60,120,240,360,720,D,M,W)
        limit (int): Количество свечей (макс. 1000)
        start_time (int): Временная метка начала в миллисекундах
        end_time (int): Временная метка окончания в миллисекундах

    Возвращает:
        list: Список свечей или None при ошибке
    """
    try:
        response = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limit,
            start=start_time,
            end=end_time
        )
        # Проверяем успешность запроса
        if response['retCode'] == 0:
            return response['result']['list']
        else:
            print(f"Ошибка API: {response['retMsg']}")
            return None

    except Exception as e:
        print(f"Ошибка соединения: {e}")
        return None
    finally:
        # Рекомендуемая задержка между запросами
        time.sleep(0.1)