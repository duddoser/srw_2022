import numpy as np
import pandas as pd

# чтение данных
data = pd.read_csv('forestfires.csv')
# 1. month - month of the year: 'jan' to 'dec'
# 2. FFMC - FFMC index from the FWI system: 18.7 to 96.20
# 3. DMC - DMC index from the FWI system: 1.1 to 291.3
# 4. DC - DC index from the FWI system: 7.9 to 860.6
# ISI - ISI index from the FWI system: 0.0 to 56.10 ---------------------
# 5. temp - temperature in Celsius degrees: 2.2 to 33.30
# 6. RH - relative humidity in %: 15.0 to 100
# 7. wind - wind speed in km/h: 0.40 to 9.40
# 8. rain - outside rain in mm/m2 : 0.0 to 6.4
# 9. area - the burned area of the forest (in ha): 0.00 to 1090.84
# (this output variable is very skewed towards 0.0, thus it may make sense to model with the logarithm transform)

# количество объектов: 517; признаков: 13
# print(data.shape)

# по данным первым и последним пяти строкам нельзя сказать, что есть объекты с NA
print(data.head())
print(data.tail())

# проверяем, что в каждой строке записаны данные
# print(data.info())

print(data.describe())  # статистический анализ числовых столбцов
print(data.corr())
