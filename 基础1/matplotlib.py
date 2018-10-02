import pandas as pd
import matplotlib.pyplot as plt

unrate = pd.read_csv('UNRATE.csv')
unrate['DATE'] = pd.to_datetime(unrate['DATE'])

fristDate = unrate.head(100)
plt.plot(fristDate['DATE'],fristDate['VALUE'])
plt.xticks(rotation=90)
plt.xlabel("year")
plt.ylabel("shiye")
plt.title("      ")
plt.show()