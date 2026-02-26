import json

# Загружаем данные из файла
with open(r'D:\Git\Home-project\1_price_list.json', 'r') as f:
    data = json.load(f)

# Выводим количество уникальных токенов
num_tokens = len(data)
print(f"Всего токенов: {num_tokens}")

# И выводим количество записей для каждого токена
for token, values in data.items():
    print(f"Токен {token} содержит {len(values)} записей")



# TODO: понять, как настроить вывод графика.
#  раньше было в strategy.main_strategy.py
# # STEP 4 - Plot trends and save for backtesting
# print("Plotting trends...")
# symbol_1 = '1000000PEIPEIUSDT'#"MATICUSDT"
# symbol_2 = '1000000MOGUSDT'#"STXUSDT"
# with open("1_price_list.json") as json_file:
#     price_data = json.load(json_file)
#     if len(price_data) > 0:
#         plot_trends(symbol_1, symbol_2, price_data)