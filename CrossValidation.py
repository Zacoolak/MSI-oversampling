import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
from scipy import stats
from sklearn.naive_bayes import GaussianNB

import Draw
DDraw = Draw.Draw


class CrossValidation():
    #RepeatedStratifiedKFold walidacja krzyżowa dla danych niezbalansowanych
    def RCV(X, y, over_sampler):
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
            #SMOTE Oversampling
            X_resampled_over_sampler, y_resampled_over_sampler = over_sampler.fit_resample(X_train, y_train)
            model.fit(X_resampled_over_sampler, y_resampled_over_sampler)
            # Balanced Accuracy Score
            predict = model.predict(X_test)
            over_sampler_scores.append(balanced_accuracy_score(y_test, predict))
            #T test
            sample_1 = X_resampled_over_sampler[y_resampled_over_sampler == 0]
            sample_2 = X_resampled_over_sampler[y_resampled_over_sampler == 1]
            over_sampler_t_test.append(stats.ttest_ind(sample_1, sample_2, equal_var=False))


        over_sampler_mean_score = np.mean(over_sampler_scores)
        over_sampler_std_score = np.std(over_sampler_scores)

        print(over_sampler)

        print("Balanced accuracy score: %.3f (%.3f)" % (over_sampler_mean_score, over_sampler_std_score))


        print("T-test for over-sampler")
        print(over_sampler_t_test)


        # Proporcje po oversamplingu SMOTE
        dataset = pd.DataFrame(y_resampled_over_sampler)
        target_count = dataset.value_counts()

        print('Class 0:', target_count[0])
        print('Class 1:', target_count[1])
        print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
        target_count.plot(kind='bar', title='After over-sampler')
        plt.show()
        DDraw.plot_2d_space(X_resampled_over_sampler, y_resampled_over_sampler, 'After over-sampler')
