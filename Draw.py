import matplotlib.pyplot as plt
import numpy as np
class Draw():
    # rysowanie
    def plot_2d_space(X, y, label='Classes'):
        colors = ['#1F77B4', '#FF7F0E']
        markers = ['o', 's']
        for l, c, m in zip(np.unique(y), colors, markers):
            plt.scatter(
                X[y == l, 0],
                X[y == l, 1],
                c=c, label=l, marker=m
            )
        plt.title(label)
        plt.legend(loc='upper right')
        plt.show()