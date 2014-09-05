import numpy as np
import pandas as pd
import numpy as np
import sys
import os
import math
import gc
import csv
import pylab as plt
from matplotlib.backends.backend_pdf import PdfPages



if __name__ == '__main__':

    step_size = 10000
    chunk_size = 1000000

    reader = pd.read_csv('train.csv', chunksize=chunk_size)

    ma_series = []
    count = 0
    for chunk in reader:
        print "Reading line", str(count*chunk_size)
        label = chunk[["Label"]]
        lab = np.array(label)
	ma_series.append(lab)
	gc.collect()
	count += 1

    ma_series = np.concatenate(ma_series)
    print ma_series.shape

    ma = pd.rolling_mean(ma_series, step_size)
    #ma = pd.rolling_window(ma_series, step_size, 'gaussian', std=10)
    print ma_series.shape, ma.shape
    #print ma

    plt.plot(ma)

    f = csv.writer(open('labelSeries2.csv','w'))
    for row in ma:
        f.writerow(row)
    plt.show() 
