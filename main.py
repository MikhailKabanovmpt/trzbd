import numpy as np

data_temperature = np.array([
    54.1, 55.3, 53.8, 56.2, 54.9, 55.7, 54.4, 56.1, 55.0, 54.6,
    95.0, 55.9, 54.2, 56.3, 55.1, 54.7, 55.4, 53.9, 56.0, 55.2,
    54.5, 55.6, 54.3, 56.4, 55.0, 54.9, 55.8, 54.0, 56.1, 54.8,
    55.5, 53.7, 56.2, 55.3, 54.4, 55.9, 54.1, 56.0, 55.1, 54.6,
    55.7, 54.2, 56.3, 55.4, 53.8, 56.1, 54.9, 55.6, 54.3, 56.2,
    99.0, 97.0, 55.0, 54.7, 55.8, 54.1, 56.4, 55.2, 54.5, 55.9,
    54.0, 56.1, 55.3, 54.8, 55.6, 54.2, 56.3, 55.1, 54.6, 55.7,
    100.0, 54.9, 55.4, 53.9, 56.0, 55.2, 54.4, 55.8, 54.1, 56.2,
    55.0, 54.7, 55.5, 54.3, 56.1, 55.3, 54.6, 55.9, 54.0, 56.0,
    55.1, 54.8, 55.6, 54.2, 56.3, 55.4, 53.8, 55.7, 54.9, 55.2
])

A = np.array([54.1, 55.3, 53.8, 56.2, 54.9, 55.7, 54.4, 56.1, 55.0,
              54.6, 55.9, 54.2, 56.3, 55.1, 54.7])


print("Анализ массива data_temperature")


mean_val = np.mean(data_temperature)
median_val = np.median(data_temperature)
std_val = np.std(data_temperature)
var_val = np.var(data_temperature)
max_val = np.max(data_temperature)
min_val = np.min(data_temperature)
range_val = max_val - min_val

print(f"Среднее значение: {mean_val:.3f}")
print(f"Медиана: {median_val:.3f}")
print(f"Стандартное отклонение: {std_val:.3f}")
print(f"Дисперсия: {var_val:.3f}")
print(f"Максимальное значение: {max_val}")
print(f"Минимальное значение: {min_val}")
print(f"Размах данных: {range_val:.3f}")

p25 = np.percentile(data_temperature, 25)
p50 = np.percentile(data_temperature, 50)
p75 = np.percentile(data_temperature, 75)
p90 = np.percentile(data_temperature, 90)
p99 = np.percentile(data_temperature, 99)

print(f"\n25-й перцентиль: {p25:.3f}")
print(f"50-й перцентиль: {p50:.3f}")
print(f"75-й перцентиль: {p75:.3f}")
print(f"90-й перцентиль: {p90:.3f}")
print(f"99-й перцентиль: {p99:.3f}")

q1 = np.percentile(data_temperature, 25)
q3 = np.percentile(data_temperature, 75)
print(f"\nПервый квартиль (Q1): {q1:.3f}")
print(f"Третий квартиль (Q3): {q3:.3f}")

iqr = q3 - q1
print(f"Межквартильный размах (IQR): {iqr:.3f}")

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = data_temperature[(data_temperature < lower_bound) | (data_temperature > upper_bound)]

print(f"\nГраницы для поиска выбросов: [{lower_bound:.3f}; {upper_bound:.3f}]")
print(f"Найденные выбросы: {outliers}")
print(f"Количество выбросов: {len(outliers)}")


print("Работа с матрицей (массив A)")


matrix = A.reshape(3, 5)
print("Матрица 3x5:")
print(matrix)

print(f"\nРазмерность матрицы (shape): {matrix.shape}")
print(f"Общее количество элементов: {matrix.size}")

row_sums = matrix.sum(axis=1)
col_sums = matrix.sum(axis=0)
print(f"\nСумма по строкам: {row_sums}")
print(f"Сумма по столбцам: {col_sums}")

matrix_T = matrix.T
print("\nТранспонированная матрица:")
print(matrix_T)

print("\nСтатистика по всем элементам матрицы:")
print(f"Среднее значение: {matrix.mean():.3f}")
print(f"Стандартное отклонение: {matrix.std():.3f}")
print(f"Дисперсия: {matrix.var():.3f}")
print(f"Первый квартиль (Q1): {np.percentile(matrix, 25):.3f}")
print(f"Второй квартиль (медиана, Q2): {np.percentile(matrix, 50):.3f}")
print(f"Третий квартиль (Q3): {np.percentile(matrix, 75):.3f}")

above_70 = data_temperature[data_temperature > 70]
print(f"\nКоличество замеров выше 70°C: {len(above_70)}")
print(f"Значения: {above_70}")

indices_above_70 = np.where(data_temperature > 70)[0]
print(f"Индексы этих замеров в массиве: {indices_above_70}")
