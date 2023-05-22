#Jan Cichy Piotr Kaczkowski
import numpy as np
import Draw
import RatioBeautifier as rb

DDraw = Draw.Draw
# Klasa odpowiada za sztuczne zwiększenie liczebności klas
class MakeImbalance():
    def make_imbalance(X, y, group, over_sampler):
        group = group   # klasa, którą trzeba powiększyć
        class_idx = (y == group)
        last_idx = np.max(np.where(y == group))
        X_class = X[class_idx]
        y_class = y[class_idx]
        num_doubling = 100
        # ratio*num_dublings = ratio po usunięciu
        for i in range(num_doubling-1):
            X_doubled = X_class.copy()
            y_doubled = y_class.copy()
            X = np.concatenate((X, X_doubled), axis=0)
            y = np.concatenate((y, y_doubled), axis=0)

        X_resampled, y_resampled = over_sampler.fit_resample(X,y)

        indices = np.where(y == group)[0]
        indices_to_del = [i for i in indices if i > last_idx]
        X = np.delete(X_resampled, indices_to_del, axis=0)
        y = np.delete(y_resampled, indices_to_del, axis=0)

        return X, y
        # Zwraca X i y po resamplingu, z usunietymi sztucznie wygenerowanymi próbkami
