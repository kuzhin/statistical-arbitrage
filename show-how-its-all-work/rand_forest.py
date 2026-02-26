import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import coint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression  # Добавленный импорт


# Загрузка исторических данных
def load_data(symbol):
    ticker = yf.Ticker(symbol)
    return ticker.history(period="1y", interval="1d")['Close']


# Загрузка данных для API3 и DASH
sym1 = "API3-USD"
sym2 = "DASH-USD"

df = pd.DataFrame({
    sym1: load_data(sym1),
    sym2: load_data(sym2)
}).dropna()

print(f"Загружено данных: {len(df)} торговых дней")
print(f"Диапазон дат: {df.index[0].strftime('%Y-%m-%d')} - {df.index[-1].strftime('%Y-%m-%d')}")

# 1. Тест на коинтеграцию
score, pvalue, _ = coint(df[sym1], df[sym2])
print(f"\nРезультаты теста на коинтеграцию:")
print(f"Статистика: {score:.3f}, p-value: {pvalue:.5f}")

# 2. Расчет коэффициента хеджирования
model = LinearRegression()
model.fit(df[[sym2]], df[sym1])
hedge_ratio = model.coef_[0]
# intercept = model.intercept_
# print(f"\nКоэффициент хеджирования: {hedge_ratio:.4f}")
# print(f"Интерсепт: {intercept:.4f}")

# 3. Расчет спреда
df['spread'] = df[sym1] - hedge_ratio * df[sym2]
# df['spread_z'] = (df['spread'] - df['spread'].mean()) / df['spread'].std()

# 4. Визуализация коинтеграции
# plt.figure(figsize=(16, 12))
# plt.suptitle(f"Анализ коинтеграции: {sym1} vs {sym2}", fontsize=16)

# График цен активов
# plt.subplot(3, 1, 1)
# plt.plot(df[sym1], 'b-', label=sym1, alpha=0.8)
# plt.plot(df[sym2] * hedge_ratio, 'r-', label=f"{sym2} * {hedge_ratio:.4f}", alpha=0.7)
# plt.title("Цены активов (нормализованные)")
# plt.legend()
# plt.grid(True)

# График спреда с каналами
# plt.subplot(3, 1, 2)
# plt.plot(df['spread'], 'g-', label="Спред", linewidth=1.5)
# plt.axhline(df['spread'].mean(), color='b', linestyle='--', label="Среднее")
# for i in [1, 1.5, 2]:
#     plt.axhline(df['spread'].mean() + i * df['spread'].std(),
#                 color='r', linestyle=':', alpha=0.7)
#     plt.axhline(df['spread'].mean() - i * df['spread'].std(),
#                 color='r', linestyle=':', alpha=0.7)

# Отметки пересечений нуля
# zero_crossings = np.where(np.diff(np.sign(df['spread'] - df['spread'].mean())))[0]
# plt.scatter(
#     df.index[zero_crossings],
#     df['spread'].iloc[zero_crossings],
#     color='black',
#     marker='x',
#     s=50,
#     label=f"Пересечения среднего ({len(zero_crossings)} раз)"
# )
#
# plt.title(f"Коинтеграционный спред (p-value: {pvalue:.5f})")
# plt.legend()
# plt.grid(True)
#
# # Распределение спреда
# plt.subplot(3, 1, 3)
# sns.histplot(df['spread'], kde=True, bins=50, color='purple')
# plt.axvline(df['spread'].mean(), color='b', linestyle='--', label="Среднее")
# plt.title("Распределение значений спреда")
# plt.xlabel("Значение спреда")
# plt.ylabel("Частота")
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()

# 5. Модель машинного обучения для прогнозирования спреда
df['lag1'] = df['spread'].shift(1)
df['lag2'] = df['spread'].shift(2)
df['lag3'] = df['spread'].shift(3)
df = df.dropna()

# Создание признаков и целевой переменной
X = df[['lag1', 'lag2', 'lag3']]
y = df['spread']

# Разделение данных
train_size = int(len(df) * 0.9)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Обучение модели
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Прогнозирование
df['predicted'] = model.predict(X)

# Оценка модели
test_pred = df['predicted'].iloc[train_size:]
r2 = r2_score(y_test, test_pred)
rmse = np.sqrt(mean_squared_error(y_test, test_pred))
print(f"\nОценка модели прогнозирования спреда:")
print(f"R^2: {r2:.4f}")
print(f"RMSE: {rmse:.6f}")

# 6. Визуализация прогнозов
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['spread'], 'g-', label="Реальный спред", alpha=0.7)
plt.plot(df.index, df['predicted'], 'b--', label="Прогноз ML", linewidth=1.5)
plt.axvline(df.index[train_size], color='r', linestyle='--', label="Начало тестовых данных")
plt.title(f"Прогнозирование коинтеграционного спреда\n(Random Forest Regressor: R^2={r2:.4f}, RMSE: {rmse:.5f})")
plt.xticks([])
# plt.xlabel("Дата")
plt.ylabel("Значение спреда")
plt.legend()
plt.grid(True)
plt.show()

# # 7. Анализ остатков
# residuals = df['spread'] - df['predicted']
# plt.figure(figsize=(12, 6))
# sns.histplot(residuals, kde=True, bins=50, color='orange')
# plt.title("Распределение ошибок прогноза")
# plt.xlabel("Ошибка прогноза")
# plt.grid(True)
# plt.show()
