import math
import numpy as np
import pandas as pd
from typing import Dict
from statistics import NormalDist


def prepare_data(data_path):
    print("Preparing data...")
    original_data = pd.read_csv(data_path, sep=',',
                                names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'],
                                usecols=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'],
                                dtype=str)
    # remove start / end spaces
    for column in ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']:
        original_data[column] = original_data[column].apply(lambda x: x.strip())
    return original_data


# Creates the training dataset with the records specified of the original dataset 11-50, 61-100, 111-150
def create_training_set(data: pd.DataFrame):
    training_subset_1 = data.iloc[10:50, :]
    training_subset_2 = data.iloc[60:100, :]
    training_subset_3 = data.iloc[110:150, :]
    subset = [training_subset_1, training_subset_2, training_subset_3]
    training_subset = pd.concat(subset)
    return training_subset


# Creates a table with the mean and variance of each parameter of each type of iris
# It will be used to calculate the gaussian likelihood probabilities
def create_training_table(data: pd.DataFrame):
    iris_class = data['class'].unique()
    iris_info = []
    df = pd.DataFrame(iris_info, columns=['class', 'attribute', 'mean', 'var'])
    for j in iris_class:
        subset_type = data[data['class'] == j]
        for i in range(len(subset_type.columns)-1):
            attr_column = subset_type.iloc[:, i]
            attribute = subset_type.columns[i]
            # Compute mean mu
            mu = attr_column.astype(float).mean()
            # Compute variance var,
            var = attr_column.astype(float).var()
            table = [j, attribute, mu, var]
            df.loc[len(df.index)] = table
    return df


# Creates the test dataset with the records specified of the original dataset 1-10, 51-60, 101-110
def create_testing_set(data: pd.DataFrame):
    testing_subset_1 = data.iloc[0:10, 0:4]
    testing_subset_2 = data.iloc[50:60, 0:4]
    testing_subset_3 = data.iloc[100:110, 0:4]
    subset = [testing_subset_1, testing_subset_2, testing_subset_3]
    testing_subset = pd.concat(subset)
    return testing_subset


# Saves in a DataFrame the original class values for the records in the test, so we can after compare the results
def create_comparator(data: pd.DataFrame):
    testing_subset_1_class = data.iloc[0:10, 4]
    testing_subset_2_class = data.iloc[50:60, 4]
    testing_subset_3_class = data.iloc[100:110, 4]
    subset_class = [testing_subset_1_class, testing_subset_2_class, testing_subset_3_class]
    testing_subset_class = pd.concat(subset_class)
    return testing_subset_class


# Computes the likelihood probabilities, taking into account the mean and var values
# obtained in the training table
def compute_likelihood_probabilities(testing_set, training_table):
    iris_class = training_table['class'].unique()
    probs_set: Dict[int, int] = {}
    probs_ver: Dict[int, int] = {}
    probs_vir: Dict[int, int] = {}
    for j in iris_class:
        subset_training_table = training_table[training_table['class'] == j]
        sl_mean = subset_training_table['mean'].iloc[0]
        sl_var = subset_training_table['var'].iloc[0]
        sw_mean = subset_training_table['mean'].iloc[1]
        sw_var = subset_training_table['var'].iloc[1]
        pl_mean = subset_training_table['mean'].iloc[2]
        pl_var = subset_training_table['var'].iloc[2]
        pw_mean = subset_training_table['mean'].iloc[3]
        pw_var = subset_training_table['var'].iloc[3]
        for i in testing_set.index:
            record = testing_set[testing_set.index == i]
            sl_prob = NormalDist(sl_mean, sl_var).pdf(float(record['sepal length']))
            sw_prob = NormalDist(sw_mean, sw_var).pdf(float(record['sepal width']))
            pl_prob = NormalDist(pl_mean, pl_var).pdf(float(record['petal length']))
            pw_prob = NormalDist(pw_mean, pw_var).pdf(float(record['petal width']))
            prob = sl_prob * sw_prob * pl_prob * pw_prob
            if j == "Iris-setosa":
                probs_set[i] = prob
            if j == "Iris-versicolor":
                probs_ver[i] = prob
            if j == "Iris-virginica":
                probs_vir[i] = prob
    probs_total = {'nTuple': probs_set.keys(), 'prob_set': probs_set.values(), 'prob_ver': probs_ver.values(),
                   'prob_vir': probs_vir.values()}
    probs_totaldf = pd.DataFrame(probs_total)
    return probs_totaldf


# For each record it multiplies the likelihood probability of each class by the prior probability of each class
# We assume equiprobables prior probabilities for all the types of classes. As there are 3 classes, prior prob =1/3
def compute_posterior_probabilities(lik_prob):
    probs_final: Dict[int, int, int, int] = {}
    prior_prob = 1/3
    for i in range(len(lik_prob['nTuple'])):
        pos = lik_prob['nTuple'].iloc[i]
        p_set_final = lik_prob['prob_set'].iloc[i] * prior_prob
        p_ver_final = lik_prob['prob_ver'].iloc[i] * prior_prob
        p_vir_final = lik_prob['prob_vir'].iloc[i] * prior_prob
        probs_final[pos] = [p_set_final, p_ver_final, p_vir_final]
    return probs_final


# Compares the 3 conditional probabilities(one for each class of plant) for each record
# It classifies the record has the class for which the record has a highest conditional probability
def compare_probabilities(final_probabilities):
    probabilities_max: Dict[int, str, int] = {}
    for i in final_probabilities.keys():
        prob1 = final_probabilities[i][0]
        prob2 = final_probabilities[i][1]
        prob3 = final_probabilities[i][2]
        lst = [prob1, prob2, prob3]
        max_prob = max(lst)
        if max_prob == prob1:
            probabilities_max[i] ="Iris-setosa"
        if max_prob == prob2:
            probabilities_max[i] ="Iris-versicolor"
        if max_prob == prob3:
            probabilities_max[i] ="Iris-virginica"
    return probabilities_max


# Computes the accuracy of the classifier by comparing the real class values and the ones
# the classifier has assigned
def compute_precision(comparator, prob_max):
    error = 0
    for i in comparator.index:
        print(comparator[i], "vs", prob_max[i])
        if comparator[i] != prob_max[i]:
            error += 1
    if error != 0:
        precision = (len(comparator)-error)/len(comparator)
        print("The classification isn't perfect, its precision is:", precision)
        return precision
    print("No errors were committed in the classification")
    return error


if __name__ == '__main__':
    data_path = "iris.data"
    original_data = prepare_data(data_path)
    training_set = create_training_set(original_data)
    test_set = create_testing_set(original_data)
    comparator = create_comparator(original_data)
    training_table = create_training_table(training_set)
    likelihood_probabilities = compute_likelihood_probabilities(test_set, training_table)
    final_probabilities = compute_posterior_probabilities(likelihood_probabilities)
    probabilities_max = compare_probabilities(final_probabilities)
    classifier_prec = compute_precision(comparator, probabilities_max)

