# Jan Cichy Piotr Kaczkowski
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler, SVMSMOTE

import RatioBeautifier as rb
import MakeImbalance

imb = MakeImbalance.MakeImbalance
class ImbResampler():
    def __init__(self, clf, resampler, ratio):
        # niezbędne arguenty do funkcji
        self.ratio = ratio
        self.clf = clf
        self.resampler = resampler
        pass

    def fit(self, X, y):
        # ustalenie która klasa jest większościowa
        #print(np.unique(y, return_counts=True))
        class_0, class_1 = rb.make_beautiful(X, y)
        if (class_0 < class_1):
            starting_ratio = (class_0 / class_1)
            group = 1  # kassa większościowa do zwiększeinia niezbalansowania
        elif (class_0 > class_1):
            group = 0
        else:
            group = 1

        if (self.ratio > 1):
            # zwiększanie niezbalansowania
            final_ratio = self.ratio
            # ! ratio*num_dublings = ratio po usunięciu!
            # dzielimy przez 10 żeby odpowiednie ratio wyszło
            ratio_help = final_ratio / 10
            # ustalenie resamplera oraz parametru ratio
            base_resampler = self.resampler.__init__(sampling_strategy=ratio_help)
            # z base_resampler imb.make_imbalance nie działa: TypeError: super(type, obj): obj must be an instance or subtype of type
            base_resampler2 = SMOTE(sampling_strategy=ratio_help)

            X_res, y_res = imb.make_imbalance(X, y, group, base_resampler2)
            #print("a", np.unique(y_res, return_counts=True))

        elif (self.ratio < 1):
            # zmniejszenie niezbalansowania
            # ustalenie resamplera oraz parametru ratio
            self.resampler.__init__(sampling_strategy=self.ratio)
            # resampling
            X_res, y_res = self.resampler.fit_resample(X, y)
            #print("a", np.unique(y_res, return_counts=True))

        self.clf.fit(X_res, y_res)
        return X_res, y_res

    def predict(self, X):
        predict = self.clf.predict(X)
        return predict
