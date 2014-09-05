import numpy as np
import pandas as pd
import numpy as np
import sys
import os
import math
import gc
import pylab as plt
from matplotlib.backends.backend_pdf import PdfPages





if __name__ == '__main__':

    step_size = 10
    chunk_size = 100

    reader = pd.read_csv('tmp.csv', chunksize=chunk_size)

    ctr_series = []
    count = 0
    for chunk in reader:
        print "Reading line", str(count*chunk_size)
        label = chunk[["Label"]]
        lab = np.array(label)
        ctr = np.zeros(chunk_size/step_size)
        j=0
        for i in range(0,chunk_size,step_size):
	    #print  lab[i:i+step_size], np.mean(lab[i:i+step_size])
            ctr[j] = np.mean(lab[i:i+step_size])
	    j += 1
	ctr_series.append(ctr)
        ma = pd.rolling_mean(label, step_size)
        #print label.shape, ma.shape
	#print label
	#print ma

        count += 1
	gc.collect()

    ctr_series = np.hstack(ctr_series)
    print ctr_series.shape

    cS = pd.Series(ctr_series)
    cS.to_csv('labelSeries.csv', index=True)
    cS.plot()
    plt.show() 
