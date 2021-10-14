from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

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

def add_to_file(file, y_test, y_pred, target_names, priors, vocab_size, num_tokens):
    file.write('******************************** MultinomialNB default values, try 1 ********************************\n')
    file.write('\nConfusion Matrix:\n')
    file.write(str(confusion_matrix(y_test, y_pred)))
    file.write('\n\nClassification Report:\n')
    file.write(str(classification_report(y_test, y_pred, target_names=bbc_train.target_names)))
    file.write('\nPrior Probability:\n')
    for name, prob in priors:
        file.write(f'{name}: {prob}\n')
    file.write(f'\nSize of vocabulary: {vocab_size} different words')
    file.write(f'\n\nNumber of word tokens in each class: {num_tokens}') #TODO:loop thru each class
    file.write(f'\n\nNumber of word tokens in total: {num_tokens}')

with open('bbc-performance.txt', 'w') as file:
    add_to_file(file, y_test, y_pred, bbc_train.target_names, priors, vocab_size, num_tokens)
