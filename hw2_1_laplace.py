import pandas as pd
import numpy as np
avg_age_dataset_n_tuples = 0.0
avg_age_dataset_n_1_tuples = 0.0


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


# Modify the dataset to obtain only the values with more than 25 years. This will be our dataset with n tuples
def dataset_n(data: pd.DataFrame):
    dataset_after_query = data[data['age'] > '25']
    return dataset_after_query


# Calculates the mean of the dataset
def avg_age_dataset(data: pd.DataFrame):
    age = data["age"]
    avg_age = age.astype(float).mean()
    return avg_age


# Obtain the dataset with n-1 tuples. We will drop one register which age is more distant from the mean,
# because it will give us the case of the maximum sensitivity
def dataset_n_1(data: pd.DataFrame):
    table = data
    age = table["age"]
    max_value = age.max()
    min_value = age.min()
    avg_value = age.astype(float).mean()
    d1 = float(max_value) - avg_value
    d2 = avg_value - float(min_value)
    if d1 > d2:
        df = data.index[data['age'] == max_value].tolist()
    else:
        df = data.index[data['age'] == min_value].tolist()
    index = df[0]
    table = table.drop(index)
    return table


# Calculates the sensitivity as the difference between the mean of the dataset with n tuples
# and the mean of the neighbor dataset with n-1 tuples
def compute_sensitivity():
    s = avg_age_dataset_n_tuples - avg_age_dataset_n_1_tuples
    print("The sensitivity is:", s)
    return s


# This method asks to write in the console the value in epsilon desired to achieve e-differential privacy
def obtain_epsilon():
    print("Insert the desired epsilon:")
    e = float(input())
    return e


# Calculates the laplace noise we will add to the real value of average age
def calculate_laplace(sen, eps):
    n = np.random.laplace(sen/eps)
    # print(n)
    return n


# Adds the laplace noise obtained to the real average value and calculates the error produced
def result(real_value, n):
    # print(n)
    r = real_value + n
    print("The average age of the records with age greater than 25 is", r)
    e = 2*pow((sensitivity/epsilon), 2)
    print("This value result ensures", epsilon, "-differential privacy")
    print("The variance of the Laplace noise is", e)
    return r


if __name__ == '__main__':
    data_path = "adult.data"
    original_data = prepare_data(data_path)
    dataset_n_tuples = dataset_n(original_data)
    avg_age_dataset_n_tuples = avg_age_dataset(dataset_n_tuples)
    print("The average age of the dataset with n tuples is", avg_age_dataset_n_tuples)
    dataset_n_1_tuples = dataset_n_1(dataset_n_tuples)
    avg_age_dataset_n_1_tuples = avg_age_dataset(dataset_n_1_tuples)
    print("The average age of the dataset with n-1 tuples is", avg_age_dataset_n_1_tuples)
    sensitivity = compute_sensitivity()
    epsilon = obtain_epsilon()
    noise = calculate_laplace(sensitivity, epsilon)
    result = result(avg_age_dataset_n_tuples, noise)
