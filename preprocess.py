import numpy as np
from CategoricalFeaturesToBayesianProbabilitiesFeaturizer import CategoricalFeaturesToBayesianProbabilitiesFeaturizer as CFBPF
import pandas as pd
import numpy as np
import sys
import os
import math
import gc
import csv
import pylab as plt
from matplotlib.backends.backend_pdf import PdfPages

def processTrainFeatures(input_file, chunk_size, cat_test_stats):

    stats = pd.DataFrame()

    reader = pd.read_csv(input_file, chunksize=chunk_size)
    count = 0 
    for chunk in reader:
	print 'Calculating mean of integer features on training set:' + str(count * chunk_size)
        int_features = chunk[["I"+str(i) for i in range(1,14)]]
        #.count() and .sum() return data series with number of non-NA observations
        if count == 0:
            stats['sum'] = int_features.sum()
            stats['count'] = int_features.count()
        else:
            stats['sum_chunk'] = int_features.sum()
            stats['count_chunk'] = int_features.count()
            stats['sum'] = stats[['sum', 'sum_chunk']].sum(axis=1)
            stats['count'] = stats[['count', 'count_chunk']].sum(axis=1)
            stats.drop(['sum_chunk', 'count_chunk'], axis=1, inplace=True)

        '''
        for i in range(0,chunk.shape[0]):
            for e in chunk.iloc[i, 15:41].values:
		if np.isnan(e) == False:
		    print int(e)
        '''
        count += 1
	gc.collect()

    stats['mean'] = stats['sum'] / stats['count']

    print("Calculating standard deviation of training integer features.....")
    reader = pd.read_csv(input_file, chunksize=chunk_size)
    count = 0 
    for chunk in reader:
        int_features = chunk[["I"+str(i) for i in range(1,14)]]
        if count == 0:
            stats['sq_sum'] = ((int_features - stats['mean']) ** 2).sum()
        else:
            stats['sq_sum_chunk'] = ((int_features - stats['mean']) ** 2).sum()
            stats['sq_sum'] = stats[['sq_sum', 'sq_sum_chunk']].sum(axis=1)
            stats.drop(['sq_sum_chunk'], axis=1, inplace=True)
        count += 1
	gc.collect()
    stats['std'] = (stats['sq_sum'] / (stats['count'] - 1)).apply(np.sqrt)
    stats.drop(['sq_sum'], axis=1, inplace=True)

    print("Adding Time of Day and Daily Temporal Mean to training set")
    day1 = np.linspace(0,1,6560000)
    day2 = np.linspace(0,1,6090000)
    day3 = np.linspace(0,1,6740000)
    day4 = np.linspace(0,1,6980000)
    day5 = np.linspace(0,1,6810000)
    day6 = np.linspace(0,1,6620000)
    day7 = np.linspace(0,1,6040000)
    tod = np.concatenate((day1,day2,day3,day4,day5,day6,day7))
    tod_ptr = 0
    print tod.shape

    chunk_size_small = chunk_size/2

    print("Normalizing integer training features....")
    reader = pd.read_csv(input_file, chunksize=chunk_size_small)
    #dtm_reader = pd.read_csv('labelSeries2.csv', chunksize=chunk_size_small)
    count = 0 
    for chunk in reader:
	print 'Writing processed training set:' + str(count * chunk_size_small)
        int_features = chunk[["I"+str(i) for i in range(1,14)]]
        norm_data = (int_features - stats['mean'])/stats['std']
        norm_data = norm_data.fillna(0)
        #print chunk['Id'].values 
        labels =  chunk[["Id", "Label"]]
        norm_data = pd.concat([labels, norm_data], axis=1)
        #Setting missing categorical values to 00000000 and those training set ids that are not present in test as ffffffff(setB) These ids are not features in the test space, but we use the average behavior in these ids to approximate the average behavior of those ids that are present in test but not in train(setC)
        cl_new = []
        for i in range(0,26): 
            cat_data = chunk[["C"+str(i+1)]]
            cat_data = cat_data.fillna('00000000')
	    cl = np.array(cat_data)
	    cl_col = []
	    for row in cl:
	        if row[0] not in cat_test_stats[i] and row[0] != '00000000':
		    cl_col.append('ffffffff')
		else:
		    cl_col.append(row[0])
	    cl_new.append(np.asarray(cl_col))
        cl_new = np.vstack(cl_new).T
        '''
        cat_data =  chunk[["C"+str(i) for i in range(1,27)]]
        cat_data = cat_data.fillna('00000000')
	cl = np.array(cat_data)
        #print cl
        cl_new = []
        for row in cl:
            cl_row = []
            for i in range(0,26):
                if row[i] not in cat_test_stats[i] and row[i] != '00000000':
		    cl_row.append('ffffffff')
		else:
		    cl_row.append(row[i])
            cl_new.append(cl_row)
        '''  

        labels = pd.DataFrame(cl_new, columns=["C"+str(i) for i in range(1,27)])
	#print labels

        norm_data = pd.concat([norm_data,labels], axis=1)

  	#Adding the time of day feature
        time_of_day = pd.DataFrame(tod[tod_ptr:tod_ptr+chunk_size_small],columns=["Tod"])
	tod_ptr += chunk_size_small
        norm_data = pd.concat([norm_data, time_of_day], axis=1)
 
        '''
        daily_temporal_mean = dtm_chunk[["Dtm"]]
        daily_temporal_mean = daily_temporal_mean.fillna(daily_temporal_mean.mean()) 
        norm_data = pd.concat([norm_data, daily_temporal_mean], axis=1)
        '''

	norm_data = norm_data.reset_index(drop=True) #Removes indexes from being printed in the csv file
	#print norm_data	
        if count == 0:
            norm_data.to_csv('norm.csv',mode='a',header=True,index=False)
        else:
            norm_data.to_csv('norm.csv',mode='a',header=False,index=False)
        count +=1
	gc.collect()

    return stats;

def getTrainSetCategories(input_file, chunk_size):
    cat_stats = []

    for i in range(0,26):
	frame = pd.DataFrame()
	cat_stats.append(frame)

    reader = pd.read_csv(input_file, chunksize=chunk_size)
    count = 0 
    for chunk in reader:
	print 'Creating dictionary of catgorical features on training set:' + str(count * chunk_size)
        #Generating histograms for categorical features
	for i in range(0,26):
	    column_name = "C"+ str(i+1)
	    cat_view = chunk[['Label',column_name]]
	    cframe = pd.DataFrame()
	    cframe['category'] = cat_view.groupby(column_name).size().index
	    cframe['value'] = cat_view.groupby(column_name).size().values
            cat_stats[i] = pd.concat([cat_stats[i],cframe])
        count +=1
	gc.collect()

    #Finish processing the categorical features	Aggregating stats over all chunks
    for i in range(0,26):
        cframe = pd.DataFrame()
	cframe['category'] = cat_stats[i].groupby('category').sum().index
	cframe['value'] = cat_stats[i].groupby('category').sum().values
        cat_stats[i] = cframe
    #Creating a dictionary for categories to find out unique ids in the test set
    cat_new_stats = []
    for i in range(0,26):
        cl = np.array(cat_stats[i])
        cat_dict = {cat[0]:int(cat[1]) for cat in cl}
        cat_new_stats.append(cat_dict)
        #print len(cat_dict.keys())
        #print cat_dict
    print "Finished building training set dictionary"

    return cat_new_stats


#@profile
def processTestFeatures(input_file, chunk_size, output_file, int_stats, cat_train_stats, norm_features, collapse_new_features):

    '''
    inputs: 
        input_file:  test csv file
        chunk_size:  granularity at which dataset is broken down into to load into memory
        int_stats:  feature wise mean and stddev calculate over the training set which is used to normalize test features.
        cat_train_stats:  list of dictionaries for each categorical feature for the training set(used to collapse new features into one category)
        collapse_new_features:  If set the new features which are not in cat_train_stats, are collapsed to a single category. If this is zero the new features are left as is
    returns
        -        
    '''
    num_test_records = 6042136
    print "Number of test records", num_test_records

    cat_stats = []
    for i in range(0,26):
	frame = pd.DataFrame()
	cat_stats.append(frame)
 
    if norm_features == 1: 
        print("Adding Time of Day and Daily Temporal Mean to test set")
        tod = np.linspace(0,1,num_test_records)
        tod_ptr = 0
        print tod.shape

    chunk_size_small = chunk_size/2 
    reader = pd.read_csv(input_file, chunksize=chunk_size_small)
    count = 0 
    for chunk in reader:
	print 'Processing test features:' + str(count * chunk_size_small)
        int_features = chunk[["I"+str(i) for i in range(1,14)]]

        if norm_features == 1: 
            #Normalizing integer test statistics
            norm_data = (int_features - int_stats['mean'])/int_stats['std']
            norm_data = norm_data.fillna(0)
        else:
            norm_data = int_features
        #print chunk['Id'].values 
        labels =  chunk[["Id"]]
        norm_data = pd.concat([labels, norm_data], axis=1)

        if collapse_new_features == 1:
            cl_new = []
            for i in range(0,26): 
                cat_data = chunk[["C"+str(i+1)]]
                cat_data = cat_data.fillna('00000000')
	        cl = np.array(cat_data)
	        cl_col = []
	        for row in cl:
	            if row[0] not in cat_train_stats[i] and row[0] != '00000000':
		        cl_col.append('ffffffff')
		    else:
		        cl_col.append(row[0])
	        cl_new.append(np.asarray(cl_col))
            cl_new = np.vstack(cl_new).T

            '''
	    cl = np.array(cat_data)
            cl_new = []
            for row in cl:
                cl_row = []
                for i in range(0,26):
                    if (row[i] not in cat_train_stats[i]) and row[i] != '00000000':
		        cl_row.append('ffffffff')
		    else:
		        cl_row.append(row[i])
                cl_new.append(cl_row)
	    '''

            labels = pd.DataFrame(cl_new, columns=["C"+str(i) for i in range(1,27)])
	    #print labels
        else:
	    labels = chunk[["C"+str(i) for i in range(1,27)]]
            labels = labels.fillna('00000000')
        norm_data = pd.concat([norm_data,labels], axis=1)

        if norm_features == 1: 
  	    #Adding the time of day feature
            time_of_day = pd.DataFrame(tod[tod_ptr:tod_ptr+chunk_size_small],columns=["Tod"])
	    tod_ptr += chunk_size_small
        else:
            time_of_day = chunk[["Tod"]]
        norm_data = pd.concat([norm_data, time_of_day], axis=1)
            

	norm_data = norm_data.reset_index(drop=True)
	#print norm_data	
        if count == 0:
            norm_data.to_csv(output_file,mode='a',header=True,index=False)
        else:
            norm_data.to_csv(output_file,mode='a',header=False,index=False)

        count +=1
	gc.collect()

def getTestSetCategories(input_file, chunk_size):
    cat_stats = []
    for i in range(0,26):
	frame = pd.DataFrame()
	cat_stats.append(frame)
    
    reader = pd.read_csv(input_file, chunksize=chunk_size)
    count = 0 
    for chunk in reader:
	print 'Collecting Test Set categories line:' + str(count * chunk_size)
        #Generating histograms for categorical features
	for i in range(0,26):
	    column_name = "C"+ str(i+1)
	    cat_view = chunk[[column_name]]
            cat_view = cat_view.fillna('00000000')
	    cframe = pd.DataFrame()
	    cframe['category'] = cat_view.groupby(column_name).size().index
	    cframe['value'] = cat_view.groupby(column_name).size().values
            cat_stats[i] = pd.concat([cat_stats[i],cframe])
        count +=1
	gc.collect()

    #Aggregating stats over all chunks
    for i in range(0,26):
        cframe = pd.DataFrame()
        cframe['category'] = cat_stats[i].groupby('category').sum().index
        cframe['value'] = cat_stats[i].groupby('category').sum().values
        cat_stats[i] = cframe
    print "Creating a dictionary for categories in the test set..."
    cat_new_stats = []
    for i in range(0,26):
        cl = np.array(cat_stats[i])
	cat_dict = {cat[0]:int(cat[1]) for cat in cl}
	cat_new_stats.append(cat_dict)
    cat_stats = cat_new_stats

    return cat_stats

def calculateOverlapTrainTest(cat_train_stats, cat_test_stats):

    unique_test_ids = []
    unique_train_ids = []

    for i in range(0,26):
        new_cats=0
	for cat in cat_test_stats[i]:
	    if cat not in cat_train_stats[i]:
	        new_cats += 1
	unique_test_ids.append(new_cats)
    
    for i in range(0,26):
        new_cats=0
	for cat in cat_train_stats[i]:
	    if cat not in cat_test_stats[i]:
	        new_cats += 1
	unique_train_ids.append(new_cats)

    print "Test data statistics "
    print "Name    TrainIDs   TestIDs    NewTestIds OldTrainIds"
    for i in range(0,26):
	print "C%d  %9d  %9d  %9d  %9d" % (i+1, len(cat_train_stats[i].keys()), len(cat_test_stats[i].keys()), unique_test_ids[i], unique_train_ids[i])

#@profile
def plot_dist(fig, pdf_pages, cat_stats):
    for i in range(0,26):
	dict_t = cat_stats[i]
	ax = fig.add_subplot(26,1,i+1)
	for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
	for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(3)
	plt.ylabel('C'+str(i+1), fontsize=5)
	
        s = pd.Series([dict_t[w] for w in sorted(dict_t, key= dict_t.get,reverse=True)]).hist(bins=100)
    plt.savefig(pdf_pages, format='pdf')

if __name__ == '__main__':

    '''
    init args:
    Training file name
    Test file name
    Chunk_size
    '''
    # input file, output file, leave the other two args the same; 
    #cfbpf = CFBPF("train.csv", "cfbpf.csv", "Label", ["C"+str(i) for i in range(1,27)] )
    # only do this when training the hash map (dict)
    #cfbpf.generateFeatureHashMap()
    # do this to featurize the data.
    #cfbpf.hashFeaturesToDenseMatrix()
    pp = PdfPages('catplots.pdf')
    training_file = "./norm.csv"
    test_file = "./norm_test_collapsed.csv"

    if os.path.isfile(test_file):
        os.remove(test_file)
    st = []
    cat_st = []


    if os.path.isfile(training_file) == False:
        cat_test_stats = getTestSetCategories(sys.argv[2],int(sys.argv[3]))

        stats = processTrainFeatures(sys.argv[1],int(sys.argv[3]),cat_test_stats)
        print "Training data statistics "
        print stats
        processTestFeatures(sys.argv[2],int(sys.argv[3]),'norm_test.csv',stats,cat_st,1,0)
    gc.collect()

    cat_train_stats = getTrainSetCategories(sys.argv[1],int(sys.argv[3]))

    processTestFeatures('norm_test.csv',int(sys.argv[3]),'norm_test_collapsed.csv',st,cat_train_stats,0,1)

    cat_test_stats = getTestSetCategories(sys.argv[2],int(sys.argv[3])/2)
    calculateOverlapTrainTest(cat_train_stats, cat_test_stats)

    print "Plotting distributions.... "
    fig1 = plt.figure(1)
    plot_dist(fig1, pp, cat_train_stats)
    fig2 = plt.figure(2)
    plot_dist(fig2, pp, cat_test_stats)

    pp.close()
    #plt.show()
