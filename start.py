import numpy as np
import pandas as pd
from kernels import SpectrumKernel
from svc import KernelSVC

#%%

Y = []

for i in range(3):
    df_X = pd.read_csv("data-challenge-kernel-methods-2024-2025/Xtr" + str(i) + ".csv")
    df_Y = pd.read_csv("data-challenge-kernel-methods-2024-2025/Ytr" + str(i) + ".csv")
    X_train_seq = list(df_X["seq"])
    y_train = np.array(df_Y["Bound"])
    y_train = np.where(y_train == 1, 1, -1)
    
    ##########
    kernel = SpectrumKernel(k = 8, X = X_train_seq)
    clf = KernelSVC(kernel = kernel, C = 1)
    clf.fit(X_train_seq, y_train)
    
    ##########
    
    df_X_test = pd.read_csv("data-challenge-kernel-methods-2024-2025/Xte" + str(i) + ".csv")
    X_test_seq = list(df_X_test["seq"])
    predictions_test = clf.predict(X_test_seq)
    predictions_test = np.where(predictions_test == 1, 1, 0)
    data = {"Id": df_X_test["Id"], "Bound": predictions_test}
    df_Y_test = pd.DataFrame(data = data)
    Y.append(df_Y_test)


Y = pd.concat(Y, axis=0)[["Id", "Bound"]]
Y.to_csv("Yte.csv", index = False)



