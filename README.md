# MSI-oversampling
Pierwsze testy zostały wykonanie na zbiorze danych liczącym 12345 próbek z imbalance ratio równym 0.999, liczności klas prezentują się w następujacy sposób (Class 0: 12333 Class 1: 12). Następnie z wykorzystaniem walidacji krzyżowej 5x2 każdyz  resamplerów został "wyszkolony"
Wyniki prezentują się w następujący sposób:
Przed oversamplingiem
![image](https://user-images.githubusercontent.com/118558953/233835525-ac690cac-55d2-4fe8-8b43-875d1f1d0a56.png)
![image](https://user-images.githubusercontent.com/118558953/233835555-5719f63c-9517-49e2-8bb9-71c2eff00f8b.png)
![image](https://user-images.githubusercontent.com/118558953/233835567-a2790740-602e-41da-ba34-b5b9843943d0.png)

Dla SMOTE
![image](https://user-images.githubusercontent.com/118558953/233835932-e1cb66e7-aac3-4620-83a6-fa8a99b0a2ad.png)
![image](https://user-images.githubusercontent.com/118558953/233835941-0a1501dd-2690-4dfb-8cd9-b045380450b3.png)
![image](https://user-images.githubusercontent.com/118558953/233835953-fd6c9b47-2ff7-4f78-9e26-052b39a7ba1b.png)

Dla randomOverSampler
![image](https://user-images.githubusercontent.com/118558953/233835997-b76e59b8-d0f3-4216-9e80-3e3cf721f8be.png)
![image](https://user-images.githubusercontent.com/118558953/233836004-d9763286-903e-419c-98f1-b46336594d6b.png)
![image](https://user-images.githubusercontent.com/118558953/233836012-b7e68048-85d1-48fc-8f01-10267872c06d.png)

Dla SVMSMOTE
![image](https://user-images.githubusercontent.com/118558953/233836043-9802b524-6ecc-49e4-8329-87a8b528927e.png)
![image](https://user-images.githubusercontent.com/118558953/233836067-e7bdbf89-ab0f-4367-be65-3cd58fa49a73.png)
![image](https://user-images.githubusercontent.com/118558953/233836078-2376e120-b0a3-4894-a78c-28dbb889c136.png)

Dla kazdego z reasmplerów sampling_strategy został ustawiony na 0.5
