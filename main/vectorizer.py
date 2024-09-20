import pickle
import re
import string
import time
import enchant
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from gensim.models import FastText
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import ParameterGrid


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
# Using FastText to train word embeddings based on my corpus data
vectorizer = FastText(vector_size=200, window=3, min_count=1, sg=0)
vectorizer.build_vocab(corpus_iterable=data['tokens'])
vectorizer.train(corpus_iterable=data['tokens'], total_examples=len(data['tokens']), epochs=10)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

'''
# Define the hyperparameter grid
param_grid = {
    'vector_size': [100, 200, 300],
    'window': [3, 5, 7],
    'min_count': [1, 5, 10],
    'sg': [0, 1]
}

# Define a function to train and evaluate FastText with specific hyperparameters
def train_evaluate_fasttext(vector_size, window, min_count, sg):
    model = FastText(sentences=data['tokens'], vector_size=vector_size, window=window, min_count=min_count, sg=sg)

    # Replace this with your evaluation metric
    # For example, cosine similarity between word vectors
    similarity_score = cosine_similarity(model.wv['word1'].reshape(1, -1), model.wv['word2'].reshape(1, -1))

    return similarity_score


# Define the hyperparameters to search
param_grid = {
    'vector_size': [100, 200, 300],
    'window': [3, 5, 7],
    'min_count': [1, 5, 10],
    'sg': [0, 1]
}

# Perform hyperparameter tuning
best_score = -1  # Initialize with a low value
best_hyperparameters = None

for params in ParameterGrid(param_grid):
    similarity_score = train_evaluate_fasttext(**params)

    # Replace this with your desired evaluation metric (e.g., accuracy, F1-score)
    # In this example, we look for the highest cosine similarity score
    if similarity_score > best_score:
        best_score = similarity_score
        best_hyperparameters = params

print("Best Hyperparameters:", best_hyperparameters)
'''