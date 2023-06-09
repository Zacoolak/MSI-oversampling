#Jan Cichy Piotr Kaczkowski
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SVMSMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import Draw
import Genarate
import CrossValidation
import ImbResampler as ImR

#Definiowanie potrzebnych funkcji
DDraw = Draw.Draw
DData = Genarate.Generate
CV = CrossValidation.CrossValidation

#zmienne
ratio = 2 # do resamplera
class_proportion = 0.99 # do make clasification

#generowanie zbioru danych
X, y = DData.generate(class_proportion)

#Wyswietlanie proporcji przed resamplingiem
dataset = pd.DataFrame(y)
target_count = dataset.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
target_count.plot(kind='bar', title='Count (target)')
#wyświetlanie w przestrzeni 2D rozkładu klas
plt.show()
DDraw.plot_2d_space(X, y, 'Starting point')

Imb = ImR.ImbResampler(clf=GaussianNB(), resampler=RandomOverSampler(), ratio=ratio)
CV.RCV(X, y, Imb)