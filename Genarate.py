#Jan Cichy Piotr Kaczkowski
import numpy as np
from sklearn.datasets import make_classification
#Klasa odpowiedzialna za generowanie danych
# x- proporcje klas
class Generate():
    def generate(x):
        n_samples = 12345
        n_features = 2
        n_redundant = 0
        n_clusters_per_class = 1
        flip_y = 0
        random_state = 1337
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_redundant=n_redundant,
                                   n_clusters_per_class=n_clusters_per_class, weights=[x], flip_y=flip_y,
                                   random_state=random_state)
        return X, y
