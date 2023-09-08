import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
dataset = pd.read_csv('hiring.csv')

# Preprocess the data
def convert_experience_to_int(word):
    word_dict = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12
    }
    if word.isdigit():
        return int(word)
    return word_dict.get(word.lower(), 0)  # use get to return 0 as default for unknown words

dataset['experience'].fillna('zero', inplace=True)
dataset['experience'] = dataset['experience'].apply(convert_experience_to_int)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)
dataset['test_score'] = dataset["test_score"].astype("int")

# Splitting into features and target
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Train a linear regression model
regressor = LinearRegression()
regressor.fit(X, y)
print("Model Score:", regressor.score(X, y))

# Save the trained model to a file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(regressor, model_file)
