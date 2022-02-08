from hw2_1_laplace import calculate_laplace, obtain_epsilon
# We import the classifier implement in section 3-a) because several functions used in the dpclassifier
# are the same that the ones we will be using here
from hw2_3_classifier import *
import math


# Creates a table with the mean and variance of each parameter of each type of iris
# We add Laplace noise to the means and vars we obtain
# It will be used to calculate the gaussian likelihood probabilities
def create_training_table(data, eps):
    iris_class = data['class'].unique()
    table = []
    df = pd.DataFrame(table, columns=['class', 'attribute', 'mean', 'var'])
    for j in iris_class:
        subset_type = data[data['class'] == j]
        for i in range(len(subset_type.columns)-1):
            attr_column = subset_type.iloc[:, i]
            attribute = subset_type.columns[i]
            n = len(attr_column)
            max_value = float(max(attr_column))
            min_value = float(min(attr_column))
            # Compute mean mu, its sensitivity and mu' with laplace noise
            mu = attr_column.astype(float).mean()
            mu_sensitivity = (max_value - min_value)/(n + 1)
            mu_laplace = mu + calculate_laplace(mu_sensitivity, eps)
            # Compute variance var, its sensitivity and var' with laplace noise
            var = attr_column.astype(float).var()
            var_sensitivity = math.sqrt(n)*mu_sensitivity
            # Ensure the variance has a positive value to avoid future problems
            var_laplace = -1
            while var_laplace < 0:
                var_laplace = var + calculate_laplace(var_sensitivity, eps)
            table = [j, attribute, mu_laplace, var_laplace]
            df.loc[len(df.index)] = table
    return df


# Calculate the prior probabilities P(cj) for each of the plant classes,
# taking into account the number of tuples with a Laplace noise
def compute_prior_probabilities(data, eps):
    iris_class = data['class'].unique()
    table = []
    df = pd.DataFrame(table, columns=['class', 'ntuples', 'probability'])
    for i in iris_class:
        class_column = data[data['class'] == i]
        ntuples = len(class_column)
        ntuples_laplace = ntuples + calculate_laplace(1, eps)
        table = [i, ntuples_laplace, 0]
        df.loc[len(df.index)] = table
    tuples = df['ntuples']
    total = tuples.sum()
    for i in range(len(tuples.index)):
        probability = tuples.loc[i]/total
        df.iloc[i, 2] = probability
    return df


# For each record it multiplies the likelihood probability of each class by the prior probability of each class
def compute_posterior_probabilities(prior_prob, lik_prob):
    probs_final: Dict[int, int, int, int] = {}
    class_prob = prior_prob['probability']
    for i in range(len(lik_prob['nTuple'])):
        pos = lik_prob['nTuple'].iloc[i]
        p_set_final = lik_prob['prob_set'].iloc[i] * class_prob[0]
        p_ver_final = lik_prob['prob_ver'].iloc[i] * class_prob[1]
        p_vir_final = lik_prob['prob_vir'].iloc[i] * class_prob[2]
        probs_final[pos] = [p_set_final, p_ver_final, p_vir_final]
    return probs_final


# Calculates the precision and recall for each class
def compute_precision_and_recall(prob_max):
    # tp = true positive, fp = false positive, fn = false negative
    tp_s = 0
    fp_s = 0
    fn_s = 0
    tp_ve = 0
    fp_ve = 0
    fn_ve = 0
    tp_vi = 0
    fp_vi = 0
    fn_vi = 0
    final = list(prob_max.values())
    setosa = final[0:10]
    versicolor = final[10:20]
    virginica = final[20:30]
    for i in range(len(setosa)):
        print(setosa[i])
        if setosa[i] == 'Iris-setosa':
            tp_s += 1
        if setosa[i] == 'Iris-versicolor':
            fn_s += 1
            fp_ve += 1
        if setosa[i] == 'Iris-virginica':
            fn_s += 1
            fp_vi += 1
    for i in range(len(versicolor)):
        print(versicolor[i])
        if versicolor[i] == 'Iris-setosa':
            fn_ve += 1
            fp_s += 1
        if versicolor[i] == 'Iris-versicolor':
            tp_ve += 1
        if versicolor[i] == 'Iris-virginica':
            fn_ve += 1
            fp_vi += 1
    for i in range(len(virginica)):
        print(virginica[i])
        if virginica[i] == 'Iris-setosa':
            fn_vi += 1
            fp_s += 1
        if virginica[i] == 'Iris-versicolor':
            fn_vi += 1
            fp_ve += 1
        if virginica[i] == 'Iris-virginica':
            tp_vi += 1

    if tp_s == 0:
        precision_set = 0
        recall_set = 0
    else:
        precision_set = tp_s / (tp_s + fp_s)
        recall_set = tp_s / (tp_s + fn_s)
    if tp_ve == 0:
        precision_ver = 0
        recall_ver = 0
    else:
        precision_ver = tp_ve / (tp_ve + fp_ve)
        recall_ver = tp_ve / (tp_ve + fn_ve)
    if tp_vi == 0:
        precision_vir = 0
        recall_vir = 0
    else:
        precision_vir = tp_vi / (tp_vi + fp_vi)
        recall_vir = tp_vi / (tp_vi + fn_vi)
    print("Class: Iris-setosa", "Precision", precision_set , "Recall", recall_set)
    print("Class: Iris-versicolor", "Precision", precision_ver, "Recall", recall_ver)
    print("Class: Iris-virginica", "Precision", precision_vir, "Recall", recall_vir)

    return precision_set, precision_ver, precision_vir, recall_set, recall_ver, recall_vir


if __name__ == '__main__':
    data_path = "iris.data"
    original_data = prepare_data(data_path)
    training_set = create_training_set(original_data)
    test_set = create_testing_set(original_data)
    comparator = create_comparator(original_data)
    epsilon = obtain_epsilon()
    training_table = create_training_table(training_set, epsilon)
    prior_probabilities = compute_prior_probabilities(training_set, epsilon)
    likelihood_probabilities = compute_likelihood_probabilities(test_set, training_table)
    final_probabilities = compute_posterior_probabilities(prior_probabilities, likelihood_probabilities)
    probabilities_max = compare_probabilities(final_probabilities)
    classifier_prec = compute_precision_and_recall(probabilities_max)
