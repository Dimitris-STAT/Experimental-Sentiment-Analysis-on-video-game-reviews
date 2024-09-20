import pickle
import re
import string
import time
import enchant
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from tensorflow import keras
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, MaxPooling1D
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Start time
start_time = time.time()
# Plot output style
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)
# Load the Spacy language model
nlp = spacy.load('en_core_web_sm')
# Load the PyEnchant spell checker
spell_checker = enchant.Dict('en_US')
# Load the spaCy English model

# Load amazon reviews dataset
path_to_feather = "part3.feather"
# Assign values to [data] pandas Dataframe
data = pd.read_feather(path_to_feather, use_threads=True)

# Equal sample for stars
data = data.groupby('stars').apply(lambda x: x.sample(n=4000)).reset_index(level=0, drop=True)

# Retain a randomized dataframe
# Frac indicates the hole set
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
print(data['stars'].value_counts())
data.to_csv('data_file.csv', index=False)

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ DATA PREPROCESS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Combined the text columns
data['reviewText'] = data['reviewText'] + data['summary']
# Drop column summary as it is with reviewText
data = data.drop(columns=['summary'], axis=1)

# float64 --> int64
data['stars'] = data['stars'].astype('int64')
data['stars'] = data['stars'].fillna(3)

# Fill None type values
data['reviewText'] = data['reviewText'].fillna('')

# Define a regular expression pattern to match extra spaces
space_pattern = re.compile(r'\s{2,}')
# Clean up extra spaces in the reviewText column
data['reviewText'] = data['reviewText'].str.replace(space_pattern, '')

# Replace uppercase with lowercase letters
data['reviewText'] = data['reviewText'].apply(lambda x: x.lower())

# Create a CountVectorizer instance to convert text into numerical vectors
count_vectorizer = CountVectorizer()

# Fit the vectorizer on the 'reviewText' column to build the vocabulary
count_vectorizer.fit(data['reviewText'])

# Transform the 'reviewText' column into numerical vectors
vectors = count_vectorizer.transform(data['reviewText'])

# Compute the cosine similarity matrix
cosine_similarities = cosine_similarity(vectors)

# List to store indices of rows to be deleted
rows_to_delete = []

# Check for duplicates or similar reviews
for i in range(len(data)):
    for j in range(i + 1, len(data)):
        similarity = cosine_similarities[i, j]
        if similarity == 1.0 and data['asin'].iloc[i] == data['asin'].iloc[j]:
            rows_to_delete.append(j)
        elif similarity > 0.9 and data['asin'].iloc[i] == data['asin'].iloc[j]:
            rows_to_delete.append(j)

# Delete the rows with duplicate or similar reviews
data = data.drop(data.index[rows_to_delete])

# Reset the index of the DataFrame
data = data.reset_index(drop=True)


# \\\\\\\\\\\\\\\\\\\\\\ DEFINE FUNCTIONS SECTION \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# Define a function to remove numbers from 'reviewText' column
def remove_numbers(text):
    return re.sub(r'\d+', '', text)


# Define punctuation removal function
def punctuation_removal(text):
    # Add an exception for the exclamation mark
    exclude = set(string.punctuation)
    exclude.remove('!')
    all_list = [char for char in text if char not in exclude]
    clean_str = ''.join(all_list)
    return clean_str


# Define a function to remove html tags inside our reviewText columns
def remove_html_tags(text):
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text(separator=" ")
    return cleaned_text


# Define a function to remove urls tags inside our reviewText columns
def remove_urls(text):
    # Remove URLs using regular expression
    cleaned_text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    return cleaned_text


# Dictionary of abbreviations
abbreviations = {"lol": "laughing out loud", "omg": "oh my God", "btw": "by the way",
                 "idk": "i do not know", "imo": "in my opinion", "tbh": "to be honest", "brb": "be right back",
                 "gtg": "got to go", "fyi": "for your information", "wtf": "what the f***",
                 "omw": "on my way", "rofl": "rolling on the floor laughing", "afaik": "as far as I know",
                 "bff": "best friends forever", "imho": "in my humble opinion", "jk": "just kidding",
                 "irl": "in real life", "tbt": "throwback Thursday", "fomo": "fear of missing out",
                 "np": "no problem", "wbu": "what about you", "ttyl": "talk to you later",
                 "gr8": "great", "srsly": "seriously", "thx": "thanks"}
# Dictionary of contractions
contractions = {"ain't": "are not", "aren't": "are not", "can't": "cannot", "could've": "could have",
                "couldn't": "could not", "didn't": "did not", "wasn't": "was not",
                "doesn't": "does not", "don't": "do not",
                "hadn't": "had not", "hasn't": "has not",
                "haven't": "have not", "he'd": "he would",
                "he'll": "he will", "he's": "he is", "i'd": "I would",
                "i'll": "I will", "i'm": "I am", "i've": "I have",
                "isn't": "is not", "it'd": "it would", "it'll": "it will",
                "it's": "it is", "let's": "let us", "im": "i am ", "ive": "i have",
                "might've": "might have", "must've": "must have",
                "mustn't": "must not", "shan't": "shall not",
                "she'd": "she would", "she'll": "she will", "she's": "she is",
                "should've": "should have", "shouldn't": "should not", "that's": "that is",
                "there's": "there is", "they'd": "they would", "they'll": "they will",
                "they're": "they are", "they've": "they have", "we'd": "we would",
                "we're": "we are", "we've": "we have", "weren't": "were not",
                "what'll": "what will", "what're": "what are", "what's": "what is",
                "what've": "what have", "where's": "where is", "who'd": "who would",
                "who'll": "who will", "who're": "who are", "who's": "who is",
                "who've": "who have", "won't": "will not", "would've": "would have",
                "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
                "you're": "you are", "you've": "you have"}


# Create a function that uses the contraction dict
def expand_contractions(text):
    pattern = re.compile("|".join(contractions.keys()), flags=re.IGNORECASE)
    expanded_text = pattern.sub(lambda match: contractions[match.group(0).lower()], text)
    return expanded_text


def expand_abbreviations(text):
    abbrev_pattern = re.compile("|".join(abbreviations.keys()), flags=re.IGNORECASE)
    expanded_txt = abbrev_pattern.sub(lambda match: abbreviations[match.group(0).lower()], text)
    return expanded_txt


# Define function to correct 3 or more consecutive letters
def correct_consecutive_letters(text):
    matches = re.findall(r'\w{3,}', text)
    for match in matches:
        corrected = re.sub(r'(\w)\1+', r'\1\1', match)
        text = text.replace(match, corrected)
    return text


# Function to find concatenated words
def find_concatenated_words(text):
    concatenated_words = []
    pattern = r'(\w)\1+'  # Matches any two or more concatenated words without space
    matches = re.findall(pattern, text)
    for match in matches:
        if ' ' not in match:
            concatenated_words.append(match)
    return concatenated_words


# Create a function that separates the concatenated words found previously
def separate_concatenated_words(text, concatenated_words):
    for match in concatenated_words:
        words = [word for word in re.split(r'([A-Z][a-z]*)', match) if word]
        if len(words) > 1:
            text = text.replace(match, ' '.join(words))
    return text


# Vectorize dataframe containing tokens
def vectorize_tokens(df, model):
    vectorized = []
    for tokens in df['tokens']:
        vec = np.zeros(model.vector_size)
        count = 0
        for word in tokens:
            if word in model.wv.key_to_index:
                vec += model.wv[word]
                count += 1
        if count > 0:
            vec /= count
        vectorized.append(vec)
    if not vectorized:
        # Return a default vector instead of None
        default_vec = np.zeros(model.vector_size)
        vectorized.append(default_vec)
    return vectorized


# Create function to remove stop words
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    negation_words = {"not", "never", "no"}  # Add more negation words as needed
    stop_words -= negation_words  # Exclude negation words from stop words
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


# \\\\\\\\\\\\\\\\\ Apply the functions from the functions section \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# Apply punctuation removal
data['reviewText'] = data['reviewText'].apply(punctuation_removal)

# Remove URLs from text
data['reviewText'] = data['reviewText'].apply(remove_urls)

# Remove html tags from text
data['reviewText'] = data['reviewText'].apply(remove_html_tags)

# Apply numeric removal
data['reviewText'] = data['reviewText'].apply(remove_numbers)

data['reviewText'] = data['reviewText'].apply(remove_stop_words)

# Apply expand_contractions to 'reviewText' column
data['reviewText'] = data['reviewText'].apply(expand_contractions)

# Apply expand_abbreviations to 'reviewText' column
data['reviewText'] = data['reviewText'].apply(expand_abbreviations)

# Apply find_concatenated_words to reviewText column
data['concatenated_words'] = data['reviewText'].apply(find_concatenated_words)

# Apply separate_concatenated_words to 'reviewText' column depending on what we find
# in the application of find_concatenated_words function!
data['reviewText'] = data.apply(
    lambda row: separate_concatenated_words(row['reviewText'], row['concatenated_words']), axis=1)
# Drop 'concatenated_words' as it is not needed anymore
data = data.drop(columns='concatenated_words')

# Apply correct_consecutive_letters to 'reviewText' column
data['reviewText'] = data['reviewText'].apply(correct_consecutive_letters)
print(data['reviewText'].head(10))

# Tokenize all the rows
data['tokens'] = data['reviewText'].apply(lambda x: word_tokenize(x))
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

min_rating = data['stars'].min()
max_rating = data['stars'].max()
data['stars'] = (data['stars'] - min_rating) / (max_rating - min_rating)
print(data['stars'].head(5))

# Now we perform calculation of weights in order to avoid biased classification afterwards #####################
data['absolute_distance'] = np.abs(data['stars'] - np.round(data['stars']))
data['class_weights'] = 1 / data['absolute_distance']
class_weights = data['class_weights'].values
data.drop(columns='absolute_distance')

# Now we map the values to the closest integers ##################################
data['stars'] = data['stars'].apply(lambda x: round(x))
print(data['stars'].head(5))

# Vectorize the tokenized 'reviewText' column
data['vectors'] = vectorize_tokens(data, vectorizer)
'''vectors = data['vectors'].tolist()
stars = data['stars'].tolist()
X_train, X_test, y_train, y_test = train_test_split(vectors, stars, test_size=0.2, random_state=42)

# Create an instance of the SVC classifier
svc = SVC()

# Define the hyperparameter grid
svc_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.1, 1, 10]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=svc, param_grid=svc_param_grid, cv=5, n_jobs=-1)

# Fit the GridSearchCV object to your data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the final model
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)

'''

########################################### CNN MODEL ####################################
texts = data['reviewText'].tolist()
labels = data['stars'].tolist()

# Tokenize with Keras Tokenizer for better synergy
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to ensure they have the same length
max_sequence_length = 200  # Adjust as needed
sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Convert the FastText word embeddings to a matrix
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, 200))  # 100 is the FastText vector size
for word, i in word_index.items():
    if word in vectorizer.wv:
        embedding_matrix[i] = vectorizer.wv[word]


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)


# Define a function to create your Keras model
def create_model(batch_size=32, epochs=10, learning_rate=0.001, dropout_rate=0.2):
    # Build a CNN model
    model = keras.Sequential()
    model.add(Embedding(len(word_index) + 1, 200, weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification, so using sigmoid activation

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Create a KerasClassifier with your model function
model = KerasClassifier(build_fn=create_model, learning_rate=0.001, dropout_rate = 0.3)


# Define the hyperparameter grid
param_grid = {
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 30],
    'learning_rate': [0.001, 0.01, 0.1],
    'dropout_rate': [0.2, 0.3, 0.4],
}

# Create the RandomizedSearchCV object
tuner = RandomizedSearchCV(estimator=model,
                                   param_distributions=param_grid,
                                   n_iter=10,  # Number of parameter settings to sample
                                   cv=3)  # Number of cross-validation folds

# Fit the tuner to your training data
tuner.fit(X_train, np.array(y_train))
best_hyperparameters = tuner.best_params_
print("\n\nBest Hyperparameters:", best_hyperparameters)










