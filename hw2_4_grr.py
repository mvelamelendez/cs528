import random
import pandas as pd
import numpy as np
from typing import Dict
from hw2_1_laplace import obtain_epsilon

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


# We define our domain as the range of ages in the dataset
def set_domain(data: pd.DataFrame):
    age = data['age']
    domain = np.sort(age.unique()).astype(int).tolist()
    return domain


# We calculate the probability p, which defines if we change the real value or not
def compute_p():
    d = len(domain)
    epsilon
    p = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
    return p


# User
def user_computation(data: pd.DataFrame):
    age = data['age']
    max_value = float(age.max())
    min_value = float(age.min())
    age_perturbed: Dict[int, int] = {}
    p = compute_p()
    for record in age.index:
        #print(record, age.iloc[record])
        true = np.random.choice([0, 1], p=[p, 1 - p])
        if true == 0:
            age_perturbed[record] = int(age.iloc[record])
        else:
            a_p = age.iloc[record]
            while a_p == age.iloc[record]:
                a_p = random.randint(min_value, max_value)
            age_perturbed[record] = a_p
    return age_perturbed


def aggregate(responses):
    a_p: Dict[int, int] = {}
    n_tuples: Dict[int, int] = {}
    ages_per = set(responses.values())
    p = compute_p()
    q = 1 - p
    n = len(responses)
    print(p,q)
    for j in ages_per:
        sum = 0
        for i in responses.keys():
            if responses[i] == j:
                sum += 1
        a_p[j] = sum
        n_tuples[j] = (sum - n * q) / (p - q)
    return list(n_tuples.values())


def grr():
    responses = user_computation(original_data)
    counts = aggregate(responses)
    for i in range(len(counts)):
        print(counts[i])
    return list(zip(domain, counts))


if __name__ == '__main__':
    data_path = "adult.data"
    original_data = prepare_data(data_path)
    epsilon = obtain_epsilon()
    domain = set_domain(original_data)
    result = grr()
