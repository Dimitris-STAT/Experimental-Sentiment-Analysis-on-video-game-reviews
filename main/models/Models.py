import pickle
import re
import string
import time
import enchant
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import xgboost as xgb
import seaborn as sns
from bs4 import BeautifulSoup
from keras.layers import Dropout
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# Start time
start_time = time.time()
# Plot output style
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)

# Load the Spacy language model
nlp = spacy.load('en_core_web_sm')


# Load amazon reviews dataset
path_to_feather = "part3.feather"
# Assign values to [data] pandas Dataframe
data = pd.read_feather(path_to_feather, use_threads=True)

# Equal sample for stars
data = data.groupby('stars').apply(lambda x: x.sample(n=1000)).reset_index(level=0, drop=True)

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
print(data['reviewText'].head(5))

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
    abbrev_pattern = re.compile(r"\b" + "|".join(abbreviations.keys()), flags=re.IGNORECASE)
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


# Now we map the values to the closest integers ##################################
data['stars'] = data['stars'].apply(lambda x: round(x))
print(data['stars'].head(5))

# Vectorize the tokenized 'reviewText' column
data['vectors'] = vectorize_tokens(data, vectorizer)
######################################### LR #######################################
# Train model to predict sentiment based on stars rating
lr = LogisticRegression(max_iter=1000)
vectors = data['vectors'].tolist()
stars = data['stars'].tolist()
X_train, X_test, y_train, y_test = train_test_split(vectors, stars, test_size=0.2, random_state=42)

y_train = np.array(y_train)
# Train the classifier with vectorized 'reviewText' and stars as labels
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

report_lr = classification_report(y_test, y_pred)
print('\nClassification Report for LogisticRegression:\n', report_lr)


# Save the Logistic Regression classifier reusable file
with open('log_reg.pkl', 'wb') as f:
    pickle.dump(lr, f)

######################################### SVM #####################################
svm_model = SVC(C=10, gamma='scale', kernel='rbf')

# Train the SVM model
svm_model.fit(X_train, y_train)


svm_y_pred = svm_model.predict(X_test)
# Calculate accuracy and print classification report

report_svm = classification_report(y_test, svm_y_pred)
print('\nClassification Report for SVM :\n', report_svm)
# Save the Support Vector Machines classifier reusable file
with open('svm.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

########################################### Random Forest ####################################

rf_classifier = RandomForestClassifier()

# Train the Random Forest Model
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

report_rf = classification_report(y_test, y_pred_rf)
print('\nClassification Report for RandomForest:\n', report_rf)
# Save the Support Vector Machines classifier reusable file
with open('rf.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)

#################################### AUC-ROC CURVE FOR Random Forests ##################################################

# Calculate the AUC-ROC score
auc_roc = roc_auc_score(y_test, y_pred_rf)

# Calculate the False Positive Rate (FPR) and True Positive Rate (TPR)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf)

# Set font style and size
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = %.2f)' % auc_roc)

# Add the diagonal line representing random chance
plt.plot([0, 1], [0, 1], 'k--')

# Customize axis labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (RF)')

# Add legend to the lower right corner
plt.legend(loc='lower right')

# Adjust spacing and layout
plt.tight_layout()

# Save the plot as a high-resolution image for inclusion in the thesis
plt.savefig('auc_roc_curve_lr.png', dpi=300)

# Show the plot (optional)
plt.show()

########################################### XGBoost ####################################
xgb_classifier = xgb.XGBClassifier()

# Train the XGBoost model
xgb_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred_xgb = xgb_classifier.predict(X_test)

report_xgb = classification_report(y_test, y_pred_xgb)
print('\n Classification Report XGBoost:\n', report_xgb)
with open('xgb.pkl', 'wb') as f:
    pickle.dump(xgb_classifier, f)

#################################### AUC-ROC CURVE FOR XGBoost ##################################################

# Calculate the AUC-ROC score
auc_roc = roc_auc_score(y_test, y_pred_xgb)

# Calculate the False Positive Rate (FPR) and True Positive Rate (TPR)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_xgb)

# Set font style and size
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = %.2f)' % auc_roc)

# Add the diagonal line representing random chance
plt.plot([0, 1], [0, 1], 'k--')

# Customize axis labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (XGB)')

# Add legend to the lower right corner
plt.legend(loc='lower right')

# Adjust spacing and layout
plt.tight_layout()

# Save the plot as a high-resolution image for inclusion in the thesis
plt.savefig('auc_roc_curve_lr.png', dpi=300)

# Show the plot (optional)
plt.show()


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

# Set hyperparameters
batch_size = 32  # Adjust as needed
learning_rate = 0.01
dropout_rate = 0.4
epochs = 10  # Adjust as needed

# Build a CNN model
model = keras.Sequential()
model.add(Embedding(len(word_index) + 1, 200, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Dropout(dropout_rate))  # Add dropout
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification, so using sigmoid activation

# Compile the model with the specified learning rate
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, np.array(y_train), batch_size=batch_size, epochs=epochs, validation_data=(X_test, np.array(y_test)))
# Evaluate the model
loss, accuracy = model.evaluate(X_test, np.array(y_test))
cnn_predict = model.predict(X_test)
binary_predictions = (cnn_predict > 0.5).astype(int)
class_report = classification_report(y_test, binary_predictions)

print("\n\nClassification Report on 'new' data for Convolutional Neural Networks\n",
      class_report)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')

#################################### CONFUSION MATRIX FOR CNN ##################################################
cm = confusion_matrix(y_test, binary_predictions)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)

# Customize axis labels and title
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')
plt.suptitle('Convolutional Neural Networks')

# Show the heatmap
plt.show()


#################################### AUC-ROC CURVE FOR CNN ##################################################
# Calculate the AUC-ROC score
auc_roc = roc_auc_score(y_test, binary_predictions)

# Calculate the False Positive Rate (FPR) and True Positive Rate (TPR)
fpr, tpr, thresholds = roc_curve(y_test, binary_predictions)

# Set font style and size
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = %.2f)' % auc_roc)

# Add the diagonal line representing random chance
plt.plot([0, 1], [0, 1], 'k--')

# Customize axis labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (CNNs)')

# Add legend to the lower right corner
plt.legend(loc='lower right')

# Adjust spacing and layout
plt.tight_layout()

# Save the plot as a high-resolution image for inclusion in the thesis
plt.savefig('auc_roc_curve_cnn.png', dpi=300)

# Show the plot (optional)
plt.show()


with open('cnn.pkl', 'wb') as f:
    pickle.dump(model, f)




# STOP CALCULATING TIME
# End time
end_time = time.time()
# Time Elapsed
elapsed_time = end_time - start_time
# Print time elapsed
print(f"\n Elapsed time: {elapsed_time: .2f} seconds")
