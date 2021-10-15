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
bbc_train = load_files('./BBC',encoding='latin1')

# Task 1.2
n, bins, patches = plt.hist(bbc_train.target, color='blue', edgecolor='black')
plt.title('Distribution of Instances in Each Class')
xticks = [0, 1, 2, 3, 4]
plt.xticks(xticks, labels=bbc_train.target_names)
plt.savefig('BBC-distribution.pdf')

# Task 1.4
#initialize a count vectorizer for the data
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(bbc_train.data)
vocab_size = len(count_vect.get_feature_names())
num_tokens = sum(sum(X_train_counts.toarray()))

class_sums = [0, 0, 0, 0, 0]
class_zeros = [0, 0, 0, 0, 0]

print(X_train_counts.get_shape())
for row, category in zip(X_train_counts.toarray(), bbc_train.target):
    class_sums[category] += row.sum()
class_sums = list(zip(bbc_train.target_names, class_sums))

ones = [x for row in X_train_counts.toarray() for x in row if x == 1]
freq_one = (sum(ones), sum(ones)/num_tokens)

#initialize a tfidf transformer for the data
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

"""
# Task 1.5 TODO: Do we use tfidf here or counts?
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, bbc_train.target, test_size=0.2, train_size=0.8, random_state=None)

# Task 1.6
clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
print(predicted)
"""
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

X_train, X_test, y_train, y_test = train_test_split(bbc_train.data, bbc_train.target, test_size=0.2, train_size=0.8, random_state=None)

text_clf.fit(X_train, y_train)

y_pred = text_clf.predict(X_test)

n = [x for x in n if x > 0]
priors = list(zip(bbc_train.target_names, [x/sum(n) for x in n]))

def add_to_file(title, file, y_test, y_pred, target_names, priors, vocab_size, num_tokens, class_sums, freq_one):
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
    file.write('\n(i) Number of words with frequency 0 in each class:\n') #TODO
    file.write('\n(j) Number of words with frequency 1 in the corpus:\n')
    file.write(f'Words: {freq_one[0]} Percentage: {freq_one[1]}\n')
    file.write(f'\n(k) Log Prob of the word \'programme\': ') #TODO
    file.write(f'\nLog Prob of the word \'laptops\': \n\n\n') #TODO

with open('bbc-performance.txt', 'w') as file:
    # Task 1.7
    title = "MultinomialNB default values, try 1"
    add_to_file(title, file, y_test, y_pred, bbc_train.target_names, priors, vocab_size, num_tokens, class_sums, freq_one)

    # Task 1.8
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])
    text_clf.fit(X_train, y_train)
    y_pred = text_clf.predict(X_test)
    title = "MultinomialNB default values, try 2"
    add_to_file(title, file, y_test, y_pred, bbc_train.target_names, priors, vocab_size, num_tokens, class_sums, freq_one)

    # Task 1.9
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(alpha=0.0001)),
    ])
    text_clf.fit(X_train, y_train)
    y_pred = text_clf.predict(X_test)
    title = "MultinomialNB default values, try 3"
    add_to_file(title, file, y_test, y_pred, bbc_train.target_names, priors, vocab_size, num_tokens, class_sums, freq_one)

    # Task 1.10
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(alpha=0.9)),
    ])
    text_clf.fit(X_train, y_train)
    y_pred = text_clf.predict(X_test)
    title = "MultinomialNB default values, try 4"
    add_to_file(title, file, y_test, y_pred, bbc_train.target_names, priors, vocab_size, num_tokens, class_sums, freq_one)
