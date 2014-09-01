__author__ = 'P Adkins'

from collections import defaultdict
import pandas as pd
import numpy as np
import gc

class CategoricalFeaturesToBayesianProbabilitiesFeaturizer(object):

    """

    CategoricalFeaturesToBayesianProbabilitiesFeaturizer( input_file,
    output_file, label_label, feature_names, **kwargs)

    This class converts a set of categorical variables to a set of
    ordinal variables by replacing them with the class conditional
    probability given the category.

    Only set up for binary classification currently.

    Made specifically to allow for more effective learning with a random
    forest.

    No information is lost if each category contains a distinct #
    of positives.

    A regularization parameter prevents low-information categories
    from polluting the outside edges of the feature values.


    """


    def __init__(self, input_file, output_file, label_label, feature_names, **kwargs):

        # Filename from which data are iteratively read

        self.input_file = input_file

        # Filename to which transformed dense matrix will be written ( as .csv )

        self.output_file = output_file

        # Label for the class label field in pd table

        self.label_label = label_label

        # List of feature names to include in hash_map

        self.feature_names = feature_names

        # Estimate of std of partition class probabilities from the
        #     overall class probability

        self.estimated_partition_density_difference = 0.01

        # Batch size in which to read .csv

        self.chunk_size = 100000

        # Prevalence threshold for feature inclusion

        self.inclusion_threshold = 100

        # Hash map; the main product of this class

        self.features_to_probabilities_hash_map = {}

        # Total number of samples found in input_file

        self.total_samples = 0

        # Total number of positive samples

        self.total_positive_samples = 0

        for k,v in kwargs.items():

            if k == "estimated_partition_density_difference":

                self.estimated_partition_density_difference = v

            elif k == "chunk_size":

                self.chunk_size = v

            elif k == "inclusion_threshold":

                self.inclusion_threshold = v

            else:

                assert False, "Unrecognized kwarg passed"



    def calculateRegularizationParam(self, u, n):

        """

        Under the assumption of a binomial distribution
            with the given mean, we should be able to calculate
            a param s.t. given the prob given the # samples used to
            calculate.

        Indeed I have several hours later performed this calculation
            and given:

        mixing param: alpha

        an estimate of the mean: u_0

        theoretical mean over partition: u

        empirical mean over partition: u_e

        number of samples in mean: n

        Estimate of mean can be either theoretical or empirical; just
            must be accurate.

        the minimum variance estimate of the mean u_var_min is
            given by alpha*(u_0) + (1-alpha)*(u_e)

        where alpha = (1 + n * (u_0-u)**2 / [u*(1-u)] )**-1

        I'll estimate u by u_0 for weak predictors (features) and
            calculate (u_0-u) by the expected value over partitions
            with plenty of samples.

        For this particular challenge, rather than compute this value
            explicitly I think I'll take a shortcut; just assume that
            u_0-u typically is around 0.01 - that any given partition
            might only have 1% more or 1% less class occurrence than
            the dataset as a whole.

        Later on, I'll estimate this value empirically.

        """

        # expected variance of partition
        part_var = self.estimated_partition_density_difference ** 2

        # theoretical variance of class distribution ( binomial with mean u )
        class_var = u*(1-u)

        # a reasonable estimate for alpha given partition means which deviate
        #      only slightly from the global mean
        alpha_estimate = 1./(1 + n * part_var / class_var )

        return alpha_estimate

    def calculateSingleFeatureStatistics(self, feature_label):

        """

        Calculate occurrence statistics for a single feature

        inputs:

            feature_label: label of the single feature for which to calc stats

        returns:

            conditional_probabilities: dict of conditional probabilities for categories

            category_counts: dict of counts for each category

        """

        conditional_probabilities_frame = pd.DataFrame()
        category_counts_frame = pd.DataFrame()

        reader = pd.read_csv(self.input_file, chunksize=self.chunk_size)

        # For printing number of iterations

        count = -1

        # For calculating the total # of samples
        #     This is necessary for assigning probabilities to null values etc

        total_samples = 0
        total_positive_samples = 0

        for chunk in reader:

            # Progress printouts

            count += 1
            print 'Reading line:' + str(count * self.chunk_size)

            # Tallying total samples

            total_samples += len(chunk)

            # Rename label column from whatever it is to "Label"

            chunk.rename(columns={self.label_label: "Label"}, inplace=True)

            # Get view with Label and specified feature

            chunk_view = chunk[[feature_label,"Label"]]

            # Increment total positive samples

            total_positive_samples += chunk_view["Label"].sum()

            # Collect conditional probabilities for this batch

            frame = pd.DataFrame()
            frame['category']= chunk_view.groupby([feature_label]).Label.mean().index
            frame['count'] = chunk_view.groupby([feature_label]).Label.mean().values
            conditional_probabilities_frame = \
                pd.concat([conditional_probabilities_frame, frame])

            # Collect counts for this batch

            frame = pd.DataFrame()
            frame['category'] = chunk_view.groupby(feature_label).size().index
            frame['count'] = chunk_view.groupby(feature_label).size().values
            category_counts_frame = \
                pd.concat([category_counts_frame, frame])

            # Force garbage collection

            gc.collect()

        # Aggregate conditional probabilities by category over all batches

        frame = pd.DataFrame()
        frame['category'] = conditional_probabilities_frame.groupby('category').mean().index
        frame['count'] = conditional_probabilities_frame.groupby('category').mean().values
        conditional_probabilities_frame = frame

        # Aggregate feature counts by category over all batches

        frame = pd.DataFrame()
        frame['category'] = category_counts_frame.groupby('category').sum().index
        frame['count'] = category_counts_frame.groupby('category').sum().values
        category_counts_frame = frame

        # Convert to numpy arrays; I'm told this two-step process will be quicker
        #     than iterating over the frame, as series object need be created
        #     each on each iteration through a data_frame using iterrows

        conditional_probabilities_frame = np.array(conditional_probabilities_frame)

        category_counts_frame = np.array(category_counts_frame)

        # Create dictionaries from category to probability/count

        conditional_probabilities = {cat_prob[0]:cat_prob[1] for cat_prob in \
                                      conditional_probabilities_frame}

        category_counts = {cat_count[0]:cat_count[1] for cat_count in \
                               category_counts_frame}

        # Save global stats

        self.total_samples = total_samples # redundant, but who cares
        self.total_positive_samples = total_positive_samples # yeppp

        return conditional_probabilities, category_counts

    def generateFeatureHashMap(self):

        for feature_name in self.feature_names:

            total_included_category_instances = 0 # for tallying total feature inclusions
            total_included_category_positive_samples = 0 # for calculating the probability for unused features
            included_categories = 0
            excluded_categories = 0

            marg_probs, counts = self.calculateSingleFeatureStatistics(feature_name)

            self.features_to_probabilities_hash_map[feature_name] = {"default":0,
                                                                     "feature":{}}

            # Iterate through map, only keeping features which pass the count threshold

            for category, prob in marg_probs.items():

                if counts[category] >= self.inclusion_threshold:

                    # TODO - redundant calculation.  factor out later
                    overall_pos_prob = self.total_positive_samples*1./self.total_samples
                    print "DEBUG 1"
                    print overall_pos_prob

                    alpha = self.calculateRegularizationParam(overall_pos_prob, counts[category])

                    regularized_prob = alpha*(overall_pos_prob) + (1-alpha)*(prob)

                    # This awful data structure is like so:
                    #     dictionary KEY on FEATURE_NAME, VALUE of FLOAT OR DICT
                    #         FLOAT is default hash value for excluded features
                    #         DICT is hash map for included features

                    self.features_to_probabilities_hash_map[feature_name]["feature"][category] = regularized_prob

                    total_included_category_instances += counts[category]
                    total_included_category_positive_samples += int(round(counts[category]*prob))

                    included_categories += 1

                else:

                    excluded_categories += 1

            # Calculate the conditional probability for all excluded features

            total_excluded_category_instances = \
                self.total_samples -total_included_category_instances

            # Calculate number of samples belonging to excluded categories

            total_excluded_category_positive_samples = \
                self.total_positive_samples - total_included_category_positive_samples

            # Calculate the conditional probability of belonging to the positive class
            #     conditioned on belonging to an excluded category

            excluded_categories_conditional_probability = \
                total_excluded_category_positive_samples*1./total_excluded_category_instances

            # Set the default value; when evaluating the hash map later, any excluded
            #      category will trigger a return of the default probability

            self.features_to_probabilities_hash_map[feature_name]["default"] = \
                excluded_categories_conditional_probability

    def hashFeaturesToDenseMatrix(self):

        # Need to read data in in batches as before, and move, one column at a time
        # through and fill in a numpy array. use get(key,default=default) to populate
        # one column at a time.  Make sure to initialize the numpy array to zeros before
        # each iteration

        # While we should almost surely calculate the stats over all the training data,
        # we should probably set a limit on the size of the numpy matrix we pull in from
        # this function.

        # We should be able to ask for the data, in dense, transformed format, in chunks
        # much like we do when we build it.

        # In fact, perhaps the best way to go is to have this not output a dense matrix
        # but instead print a dense matrix to .csv, which we can load in batches easily
        # later with pandas to build the ensemble of random forests

        f = open(self.output_file,"wb")

        # Write header

        print >> f, ",".join(self.feature_names)

        out_matrix = np.zeros((self.chunk_size,len(self.feature_names)))

        reader = pd.read_csv(self.input_file, chunksize=self.chunk_size)

        for chunk in reader:

            # Go down one column at a time

            for column, feature in enumerate(self.feature_names):

                frame_features = list(chunk[feature])

                single_feature_hasher = self.features_to_probabilities_hash_map[feature]["feature"]

                excluded_feature_default = self.features_to_probabilities_hash_map[feature]["default"]

                # Fill numpy buffer with hashed feature values

                for row, ff in enumerate(frame_features):

                    out_matrix[row,column] = single_feature_hasher.get(ff,excluded_feature_default)

                # Write numpy buffer to .csv.  len is necessary because last chunk
                #     may not fill buffer completely

            for row in out_matrix[:len(chunk),:]:

                print >> f, ",".join(row.astype(str))

        f.close()

        pass
