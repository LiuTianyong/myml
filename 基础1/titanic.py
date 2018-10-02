import pandas as pd
import numpy as np

#读入数据
titanic_survival = pd.read_csv('titanic_train.csv')
#
# age = titanic_survival['Age']
# print(age[0:10])
# age_is_null = pd.isnull(age)
# print(age_is_null)
#
# age_null_true = age[age_is_null]
# print(age_null_true)
# age_null_count = len(age_null_true)
# print(age_null_count)

passenger_survival = titanic_survival.pivot_table(index="Embarked", values=[ "Fare", "Survived"], aggfunc=np.sum)
print(passenger_survival)







