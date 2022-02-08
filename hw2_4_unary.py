import pandas as pd
import numpy as np
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
    epsilon
    p = np.exp(epsilon/2) / (np.exp(epsilon/2) + 1)
    return p


# We encode the response by creating an array for each age in the domain
# This array is full of zeros except in the position where
def encode(response):
    return [1 if d == response else 0 for d in domain]


def perturb(encoded_response):
    return [perturb_bit(b) for b in encoded_response]


def perturb_bit(bit):
    p = compute_p()
    q = 1 - p
    sample = np.random.random()
    if bit == 1:
        if sample <= p:
            return 1
        else:
            return 0
    elif bit == 0:
        if sample <= q:
            return 1
        else:
            return 0


def aggregate(responses):
    p = compute_p()
    q = 1 - p
    sums = np.sum(responses, axis=0)
    n = len(responses)
    return [(v - n * q) / (p - q) for v in sums]


def unary_code():
    responses = [perturb(encode(r)) for r in original_data['age']]
    counts = aggregate(responses)
    for i in range(len(counts)):
        print(counts[i])
    return list(zip(domain, counts))


if __name__ == '__main__':
    data_path = "adult.data"
    original_data = prepare_data(data_path)
    #epsilon = obtain_epsilon()
    epsilon = 2
    domain = set_domain(original_data)
    result = unary_code()

