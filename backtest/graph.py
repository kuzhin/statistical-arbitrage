

# TODO:Граф должен был быть вы strategy.main_strategy
#  разобраться, как сделать тут.
# STEP 4 - Plot trends and save for backtesting
print("Plotting trends...")
symbol_1 = '1000000PEIPEIUSDT  '# "MATICUSDT"
symbol_2 = '1000000MOGUSDT  '# "STXUSDT"
with open("1_price_list.json") as json_file:
    price_data = json.load(json_file)
    if len(price_data) > 0:
        plot_trends(symbol_1, symbol_2, price_data)
