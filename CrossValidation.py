#Jan Cichy Piotr Kaczkowski
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
from scipy import stats
from sklearn.naive_bayes import GaussianNB

import Draw
import MakeImbalance
DDraw = Draw.Draw
#MakeImb = MakeImbalance.MakeImbalance

# Klasa odpowiedzialna za walidację krzyżową, obliczenie Balanced Accuracy Score, oraz wykonaanie testu T-studenta
class CrossValidation():
    # RepeatedStratifiedKFold walidacja krzyżowa dla danych niezbalansowanych
    def RCV(X, y, over_sampler):
        # Zwiększenie liczebności klasy 0 lub 1 (parametr group)
        #X, y = MakeImb.make_imbalance(X, y, 1, over_sampler)

        # Walidacja krzyżowa 5x2
        n_splits = 2
        n_repeats = 5
        random_state = 1337
        over_sampler_scores = []
        over_sampler_t_test = []
        model = GaussianNB()
        rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

        for train_index, test_index in rkf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # SMOTE Oversampling
            X_resampled_over_sampler, y_resampled_over_sampler = over_sampler.fit(X_train, y_train)
            model.fit(X_resampled_over_sampler, y_resampled_over_sampler)
            # Balanced Accuracy Score
            predict = model.predict(X_test)
            over_sampler_scores.append(balanced_accuracy_score(y_test, predict))
            # T test
            # over_sampler_t_test.append(stats.ttest_ind(X_resampled_over_sampler, y_resampled_over_sampler, equal_var=False))

        #T test, p value
        """
        over_sampler_t_test = np.zeros((len(over_sampler_scores), len(over_sampler_scores)))
        over_sampler_p_value = np.zeros((len(over_sampler_scores), len(over_sampler_scores)))
        better_values = np.zeros((len(over_sampler_scores), len(over_sampler_scores))).astype(bool)
        alpha = 0.05

        for i in range(len(over_sampler_scores)):
            for j in range(len(over_sampler_scores)):
                a = over_sampler_scores[:, i]
                b = over_sampler_scores[:, j]

                over_sampler_t_test[i, j], over_sampler_p_value[i, j] = stats.ttest_rel(a, b)
                better_values[i, j] = np.mean(a) > np.mean(b)
        """
        over_sampler_t_test, over_sampler_p_value = stats.ttest_rel(over_sampler_scores[0], over_sampler_scores[1])

        # Sprawdzenie istotności statystycznej
        alpha = 0.05
        is_relevant = over_sampler_p_value < alpha
        # Przewaga istotna statystycznie
        #is_relevant_adventage = over_sampler_p_value * better_values

        # Średnia z Balanced Accuracy
        over_sampler_mean_score = np.mean(over_sampler_scores)
        # Odchylenie standartdowe z Balanced Accuracy
        over_sampler_std_score = np.std(over_sampler_scores)

        print(over_sampler)

        print("Balanced accuracy score: %.3f (%.3f)" % (over_sampler_mean_score, over_sampler_std_score))

        print("T-test for over-sampler: \n", over_sampler_t_test)
        print("p-value for over-sampler: \n", over_sampler_p_value)
        print("Statistically significant: \n", is_relevant)
        #print("Statistically significant advantage: \n", is_relevant_adventage)


        # Proporcje po oversamplingu SMOTE
        dataset = pd.DataFrame(y_resampled_over_sampler)
        target_count = dataset.value_counts()
        print('Class 0:', target_count[0])
        print('Class 1:', target_count[1])
        print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
        target_count.plot(kind='bar', title='After over-sampler')
        # wyświetlanie w przestrzeni 2D rozkładu klas po oversamplingu
        plt.show()
        DDraw.plot_2d_space(X_resampled_over_sampler, y_resampled_over_sampler, 'After over-sampler')
