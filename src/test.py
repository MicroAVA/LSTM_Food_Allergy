import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
    df = pd.DataFrame([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    df = df.T
    print(df)

    scaler = MinMaxScaler()
    df = scaler.fit_transform(df[df.columns])
    print(df)

