#Jan Cichy Piotr Kaczkowski
import pandas as pd

# proporcje klas w postaci 1: coś
# lub
# coś : 1

def make_beautiful(X, y):

    dataset = pd.DataFrame(y)
    target_count = dataset.value_counts()
    class_0 = target_count[0]
    class_1 = target_count[1]

    ratio_0 = round(class_0/class_1, 2)
    ratio_1 = 1
    if(ratio_0 < 1):
        multiple = 1 / ratio_0
        ratio_0 = (ratio_0 * multiple)
        ratio_1 = round(1*multiple, 2)
    """
    print('Class 0:', class_0)
    print('Class 1:', class_1)
    print('Proportion:', ratio_0, ':', ratio_1)
    """
    return ratio_0, ratio_1

