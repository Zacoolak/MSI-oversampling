#Jan Cichy Piotr Kaczkowski
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
from scipy import stats
from sklearn.naive_bayes import GaussianNB
from sklearn import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import Draw
DDraw = Draw.Draw

# Klasa odpowiedzialna za walidację krzyżową, obliczenie Balanced Accuracy Score, oraz wykonaanie testu T-studenta
class CrossValidation():
    # RepeatedStratifiedKFold walidacja krzyżowa dla danych niezbalansowanych
    def RCV(X, y, over_sampler):
        clfs = [
            GaussianNB(),
            KNeighborsClassifier(),
            DecisionTreeClassifier()
        ]
        # Walidacja krzyżowa 5x2
        n_splits = 2
        n_repeats = 5
        random_state = 1337
        over_sampler_scores = np.zeros((n_splits*n_repeats, len(clfs)))

        rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

        for fold_id, (train_index, test_index) in enumerate(rkf.split(X, y)):
            # SMOTE Oversampling
            for clf_id, clf in enumerate(clfs):
                clfc = clone(clf)
                X_resampled_over_sampler, y_resampled_over_sampler = over_sampler.fit(X[train_index], y[train_index])
                clfc.fit(X_resampled_over_sampler, y_resampled_over_sampler)
                # Balanced Accuracy Score
                predict = clfc.predict(X[test_index])
                over_sampler_scores[fold_id, clf_id] = balanced_accuracy_score(y[test_index], predict)
                print(clf)
                print(over_sampler_scores)
        #print(clfc)
        # Proporcje po oversamplingu

        dataset = pd.DataFrame(y_resampled_over_sampler)
        target_count = dataset.value_counts()
        print('Class 0:', target_count[0])
        print('Class 1:', target_count[1])
        print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
        target_count.plot(kind='bar', title='After over-sampler')
        # wyświetlanie w przestrzeni 2D rozkładu klas po oversamplingu
        plt.show()
        DDraw.plot_2d_space(X_resampled_over_sampler, y_resampled_over_sampler, 'After over-sampler')

        #T test, p value

        over_sampler_t_test = np.zeros((len(clfs), len(clfs)))
        over_sampler_p_value = np.zeros((len(clfs), len(clfs)))
        better_values = np.zeros((len(clfs), len(clfs))).astype(bool)
        alpha = 0.05

        for i in range(len(clfs)):
            for j in range(len(clfs)):
                a = over_sampler_scores[:, i]
                b = over_sampler_scores[:, j]

                over_sampler_t_test[i, j], over_sampler_p_value[i, j] = stats.ttest_rel(a, b)
                better_values[i, j] = np.mean(a) > np.mean(b)

        # Sprawdzenie istotności statystycznej
        is_relevant = over_sampler_p_value < alpha
        # Przewaga istotna statystycznie
        is_relevant_adventage = better_values * is_relevant
        # Średnia z Balanced Accuracy
        over_sampler_mean_score = np.mean(over_sampler_scores)
        # Odchylenie standartdowe z Balanced Accuracy
        over_sampler_std_score = np.std(over_sampler_scores)

        print("\nBalanced accuracy score: %.3f (%.3f)" % (over_sampler_mean_score, over_sampler_std_score))
        print("\nT-test for over-sampler: \n", over_sampler_t_test)
        print("\np-value for over-sampler: \n", over_sampler_p_value)
        print("\nBetter values for clasiffiers: \n", better_values)
        print("\nStatistically significant: \n", is_relevant)
        print("\nStatistically significant advantage: \n", is_relevant_adventage)
        print()
        for clfn1, better1 in zip(clfs, is_relevant_adventage):
            for clfn2, better2 in zip(clfs, better1):
                if better2:
                    avg1 = np.mean(over_sampler_scores[:, clfs.index(clfn1)])
                    avg2 = np.mean(over_sampler_scores[:, clfs.index(clfn2)])
                    print(f"{clfn1} with {avg1:.3f} better than {clfn2} with {avg2:.3f}")
        print()