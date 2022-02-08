import pandas as pd
import numpy as np
from mpmath import *


def prepare_data(data_path):
    print("Preparing data...")
    original_data = pd.read_csv(data_path, sep=',',
                                names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                       'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                                       'salary'],
                                usecols=['age', 'education', 'marital-status', 'race', 'occupation',
                                         'salary'],
                                dtype=str)
    # remove the columns we are not going to use
    original_data = original_data[['age', 'education', 'marital-status', 'race', 'occupation', 'salary']]

    # remove start / end spaces
    for column in ['education', 'marital-status', 'race', 'occupation', 'salary']:
        original_data[column] = original_data[column].apply(lambda x: x.strip())
    return original_data


# u(x,r) is the score function that counts how many times each education level appears
def u(x: pd.DataFrame, r):
    ntuples = (x.education.values == r).sum()
    return ntuples


# Computes the exponential probabilities for each Education, normalize them and depending on those probabilities,
# returns the most frequent Education
def exponential(x: pd.DataFrame):
    R = x.education.unique()
    # Calculate the score for each element of R
    scores = [u(x, r) for r in R]
    # The sensitivity obtained is 1/ Explained in 2-b)
    sensitivity = 1
    # Obtain the desired epsilon
    print("Insert the desired epsilon:")
    epsilon = float(input())
    # Calculate the probability for each element, based on its score
    # Use mp because we obtain very big numbers
    mp.dps = 10
    probabilities = [mp.exp(epsilon * score / (2 * sensitivity)) for score in scores]
    # Normalize the probabilities so they sum to 1
    total_prob = fsum(probabilities)
    for i in range(len(probabilities)):
        probabilities[i] = fdiv(probabilities[i], total_prob)
    # List of the different Education and their normalized probabilities
    pr = list(zip(R, probabilities))
    # Choose an element from R based on the probabilities
    print("The most frequent education is", np.random.choice(R, 1, p=probabilities)[0])
    print("This value result ensures", epsilon, "-differential privacy")
    return pr, np.random.choice(R, 1, p=probabilities)[0]


if __name__ == '__main__':
    data_path = "adult.data"
    original_data = prepare_data(data_path)
    pr, result = exponential(original_data)
