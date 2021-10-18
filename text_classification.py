from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Task 1.3
#load the BBC text data into a training set
path = input("Enter the path to the directory of the dataset you would like to analyze: (Ex: \'BBC\')\n")
try:
    bbc_train = load_files(path,encoding='latin1')
except:
    print("File path was not valid.")
    exit(0)
# Task 1.2
n, bins, patches = plt.hist(bbc_train.target, color='blue', edgecolor='black')
plt.title('Distribution of Instances in Each Class')
xticks = [0] * len(bbc_train.target_names)
for i in range(len(xticks)):
    xticks[i] = i
plt.xticks(xticks, labels=bbc_train.target_names)
plt.savefig('BBC-distribution.pdf')

# Task 1.4
#initialize a count vectorizer for the data
count_vect = CountVectorizer()

# Task 1.5 Split dataset into 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(bbc_train.data, bbc_train.target, test_size=0.2, train_size=0.8, random_state=None)
X_train_counts = count_vect.fit_transform(X_train)

# Task 1.6 Train a multinomial Naive Bayes Classifier on the training set
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
text_clf.fit(X_train, y_train)
y_pred = text_clf.predict(X_test)

# The following section is used to help answer questions 1.7.e, 1.7.f, 1.7.i, 1.7.j, 1.7.k
vocab_size = len(count_vect.get_feature_names())
# Get tokens in entire corpus (train and test set)
num_tokens = sum(sum(X_train_counts.toarray())) + sum(sum(count_vect.fit_transform(X_test).toarray())) 
print(num_tokens)
class_sums = [0] * len(bbc_train.target_names)
class_zeros = [0] * len(bbc_train.target_names)
class_docs = [0, 0, 0, 0, 0]
for x in y_train:
    class_docs[x] += 1
print(len(y_train))
priors = list(zip(bbc_train.target_names, [x/len(y_train) for x in class_docs]))

# The indices of each of the inner lists corresponds to the indices of the list count_vect.get_feature_names()
class_frequencies = [
    [0] * vocab_size,
    [0] * vocab_size,
    [0] * vocab_size,
    [0] * vocab_size,
    [0] * vocab_size
]
corpus_frequencies = [0] * vocab_size
i = 0
# Add up the frequencies for each word based on class and whole corpus
print('Processing lots of data, this may take a while...')
for row, category in zip(X_train_counts.toarray(), y_train):
    for index, word_freq in enumerate(row):
        class_frequencies[category][index] += word_freq
        corpus_frequencies[index] += word_freq
    class_sums[category] += row.sum()
class_sums = list(zip(bbc_train.target_names, class_sums))

num_ones =  sum([1 for freq in corpus_frequencies if freq == 1])
freq_one = (num_ones, num_ones/num_tokens)

# Writes different statistics about the model to a file
def add_to_file(title, file, y_test, y_pred):
    file.write(f'(a) ******************************** {title} ********************************\n')
    file.write('\n(b) Confusion Matrix:\n')
    file.write(str(confusion_matrix(y_test, y_pred)))
    file.write('\n\n(c)(d) Classification Report:\n')
    file.write(str(classification_report(y_test, y_pred, target_names=bbc_train.target_names)))
    file.write('\n(e) Prior Probability:\n')
    for name, prob in priors:
        file.write(f'{name}: {prob}\n')
    file.write(f'\n(f) Size of vocabulary: {vocab_size} different words')
    file.write('\n\n(g) Number of word tokens in each class:\n')
    for name, num in class_sums:
        file.write(f'{name}: {num}\n')
    file.write(f'\n(h) Number of word tokens in total: {num_tokens}\n')
    file.write('\n(i) Number of words with frequency 0 in each class:\n')
    i = 0
    for name, num in class_sums:
        file.write(f'{name}:\n')
        num_zeros =  sum([1 for freq in class_frequencies[i] if freq == 0])
        freq_zeros = (num_zeros, num_zeros/num)
        file.write(f'words: {freq_zeros[0]} percentage: {freq_zeros[1]}\n')
        i += 1
    file.write('\n(j) Number of words with frequency 1 in the corpus:\n')
    file.write(f'words: {freq_one[0]} percentage: {freq_one[1]}\n')
    file.write(f'\n(k) Log Prob of the word \'programme\':\n')
    logProb = np.log(corpus_frequencies[count_vect.get_feature_names().index('programme')]/num_tokens)/np.log(2)
    file.write(f'{str(logProb)}')
    file.write(f'\nLog Prob of the word \'laptops\':\n')
    logProb = np.log(corpus_frequencies[count_vect.get_feature_names().index('laptops')]/num_tokens)/np.log(2)
    file.write(f'{str(logProb)}\n\n\n')

# Open bbc-performance.txt to write to
with open('bbc-performance.txt', 'w') as file:
    # Task 1.7
    title = "MultinomialNB default values, try 1"
    print(f'Writing to file, {title}')
    add_to_file(title, file, y_test, y_pred)

    # Task 1.8
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])
    text_clf.fit(X_train, y_train)
    y_pred = text_clf.predict(X_test)
    title = "MultinomialNB default values, try 2"
    print(f'Writing to file, {title}')
    add_to_file(title, file, y_test, y_pred)

    # Task 1.9
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(alpha=0.0001)),
    ])
    text_clf.fit(X_train, y_train)
    y_pred = text_clf.predict(X_test)
    title = "MultinomialNB default values, try 3"
    print(f'Writing to file, {title}')
    add_to_file(title, file, y_test, y_pred)

    # Task 1.10
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(alpha=0.9)),
    ])
    text_clf.fit(X_train, y_train)
    y_pred = text_clf.predict(X_test)
    title = "MultinomialNB default values, try 4"
    print(f'Writing to file, {title}')
    add_to_file(title, file, y_test, y_pred)
    print('Finished!')
