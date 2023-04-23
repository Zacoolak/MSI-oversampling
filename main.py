import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SVMSMOTE


import Draw
import Genarate
import CrossValidation



DDraw = Draw.Draw
DData = Genarate.Generate
CV = CrossValidation.CrossValidation

#zmienne
ratio = 0.5 # do resamplera
class_proportion = 0.999 # do make clasification
#generowanie zbioru danych
X, y = DData.generate(class_proportion)
#Wyswietlanie proporcji przed resamplingiem
dataset = pd.DataFrame(y)
target_count = dataset.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
target_count.plot(kind='bar', title='Count (target)')
plt.show()
DDraw.plot_2d_space(X, y, 'Starting point')

smote = SMOTE(sampling_strategy=ratio)

ROS = RandomOverSampler(sampling_strategy=ratio)


SVMSMOTE = SVMSMOTE(sampling_strategy=ratio)

CV.RCV(X, y, smote)
CV.RCV(X, y, ROS)
CV.RCV(X, y, SVMSMOTE)

