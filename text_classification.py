from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_files

categories = ['business', 'entertainment', 'politics', 'sport', 'tech']

bbc_train = load_files('./BBC')
print(bbc_train.target_names)
print(len(bbc_train.data))
"""count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform()

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])"""