import numpy as np
from CategoricalFeaturesToBayesianProbabilitiesFeaturizer import CategoricalFeaturesToBayesianProbabilitiesFeaturizer as CFBPF
import pandas as pd
import numpy as np
import sys
import gc
import pylab as plt
from matplotlib.backends.backend_pdf import PdfPages

def processTrainFeatures(input_file, chunk_size, norm_features, gen_stats):

    stats = pd.DataFrame()
    cat_stats = []

    for i in range(0,26):
	frame = pd.DataFrame()
	cat_stats.append(frame)

    reader = pd.read_csv(input_file, chunksize=chunk_size)
    count = 0 
    for chunk in reader:
	print 'Reading line:' + str(count * chunk_size)
        int_features = chunk.iloc[:, 2:15]
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

        #Generating histograms for categorical features
	if gen_stats == 1:
	    for i in range(0,26):
	        column_name = "C"+ str(i+1)
	        cat_view = chunk[['Label',column_name]]
	        cframe = pd.DataFrame()
	        cframe['category'] = cat_view.groupby(column_name).size().index
	        cframe['value'] = cat_view.groupby(column_name).size().values
                cat_stats[i] = pd.concat([cat_stats[i],cframe])
        '''
        for i in range(0,chunk.shape[0]):
            for e in chunk.iloc[i, 15:41].values:
		if np.isnan(e) == False:
		    print int(e)
        '''
        count += 1
	gc.collect()

    stats['mean'] = stats['sum'] / stats['count']

    #Finish processing the categorical features	
    if gen_stats == 1:
	#Aggregating stats over all chunks
        for i in range(0,26):
	    cframe = pd.DataFrame()
	    cframe['category'] = cat_stats[i].groupby('category').sum().index
	    cframe['value'] = cat_stats[i].groupby('category').sum().values
            cat_stats[i] = cframe
        #Creating a dictionary for categories to find out uniqe ids in the test set
        cat_new_stats = []
        for i in range(0,26):
	    cl = np.array(cat_stats[i])
	    cat_dict = {cat[0]:int(cat[1]) for cat in cl}
	    cat_new_stats.append(cat_dict)
	    #print len(cat_dict.keys())
	    #print cat_dict
	cat_stats = cat_new_stats


    if norm_features == 0:
        return stats,cat_stats;

    #Calculating standard deviation
    reader = pd.read_csv(input_file, chunksize=chunk_size)
    count = 0 
    for chunk in reader:
        int_features = chunk.iloc[:, 2:15]
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

    #Normalizing features
    reader = pd.read_csv(input_file, chunksize=chunk_size)
    count = 0 
    for chunk in reader:
        int_features = chunk.iloc[:, 2:15]
        norm_data = (int_features - stats['mean'])/stats['std']
        #print chunk['Id'].values 
        labels =  chunk.iloc[:, 0:2]
        norm_data = pd.concat([labels, norm_data], axis=1)
        labels =  chunk.iloc[:, 15:41]
        norm_data = pd.concat([norm_data,labels], axis=1)
	norm_data = norm_data.reset_index(drop=True)
	#print norm_data	
        if count == 0:
            norm_data.to_csv('norm.csv',mode='a',header=True,index=False)
        else:
            norm_data.to_csv('norm.csv',mode='a',header=False,index=False)
        count +=1
	gc.collect()

    #print stats,cat_stats
    return stats,cat_stats;


def processTestFeatures(input_file, chunk_size, int_stats, cat_train_stats, gen_stats):

    cat_stats = []
    for i in range(0,26):
	frame = pd.DataFrame()
	cat_stats.append(frame)
    
    reader = pd.read_csv(input_file, chunksize=chunk_size)
    count = 0 
    for chunk in reader:
	print 'Reading line:' + str(count * chunk_size)
        int_features = chunk.iloc[:, 1:14]
        #Normalizing integer test statistics
        norm_data = (int_features - int_stats['mean'])/int_stats['std']
        #print chunk['Id'].values 
        labels =  chunk.iloc[:, 0:1]
        norm_data = pd.concat([labels, norm_data], axis=1)
        labels =  chunk.iloc[:, 14:40]
        norm_data = pd.concat([norm_data,labels], axis=1)
	norm_data = norm_data.reset_index(drop=True)
	#print norm_data	
        if count == 0:
            norm_data.to_csv('norm_test.csv',mode='a',header=True,index=False)
        else:
            norm_data.to_csv('norm_test.csv',mode='a',header=False,index=False)

        #Generating histograms for categorical features
	for i in range(0,26):
	    column_name = "C"+ str(i+1)
	    cat_view = chunk[[column_name]]
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
    #Creating a dictionary for categories to find out uniqe ids in the test set
    cat_new_stats = []
    new_ids = []
    for i in range(0,26):
        cl = np.array(cat_stats[i])
        new_cats=0
	for cat in cl:
	    if cat[0] not in cat_train_stats[i].keys():
	        new_cats += 1
	cat_dict = {cat[0]:int(cat[1]) for cat in cl}
	cat_new_stats.append(cat_dict)
	#print len(cat_dict.keys())
	#print cat_dict
	new_ids.append(new_cats)
    cat_stats = cat_new_stats

    return cat_stats,new_ids

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

    stats, cat_train_stats = processTrainFeatures(sys.argv[1],int(sys.argv[3]),1,1)
    print "Training data statistics "
    print stats
    plt.figure(1)
    for i in range(0,26):
	dict_t = cat_train_stats[i]
	plt.subplot(2,13,i+1)
        s = pd.Series([dict_t[w] for w in sorted(dict_t, key= dict_t.get,reverse=True)]).hist(bins=100)
    plt.savefig(pp, format='pdf')

    cat_test_stats, new_ids = processTestFeatures(sys.argv[2],int(sys.argv[3]),stats,cat_train_stats,True)
    plt.figure(2)
    for i in range(0,26):
	dict_t = cat_test_stats[i]
	plt.subplot(2,13,i+1)
        s = pd.Series([dict_t[w] for w in sorted(dict_t, key= dict_t.get,reverse=True)]).hist(bins=100)
    plt.savefig(pp, format='pdf')

    print "Test data statistics "
    print "Name    TrainIDs   TestIDs    NewIds"
    for i in range(0,26):
	print "C%d  %5d  %5d  %5d" % (i+1, len(cat_train_stats[i].keys()), len(cat_test_stats[i].keys()), new_ids[i])


    pp.close()
    #plt.show()
