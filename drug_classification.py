import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

warnings.filterwarnings('ignore')

# Task 2.2 load csv file
drug_df = pd.read_csv('./drug200.csv')

# Task 2.3 plot class distribution
plt.hist(drug_df['Drug'])
plt.title('Distribution of Instances in Each Class')
plt.savefig('drug-distribution.pdf')

# Task 2.4 Convert ordinal and nominal features to numerical
drug_df = pd.get_dummies(drug_df, prefix=['Sex'], columns=['Sex'])
drug_df['BP'] = pd.Categorical(drug_df['BP'], ordered=True, categories=['LOW', 'NORMAL', 'HIGH']).codes
drug_df['Cholesterol'] = pd.Categorical(drug_df['Cholesterol'], ordered=True, categories=['NORMAL', 'HIGH']).codes

# Task 2.5 split the dataset
X_train, X_test, y_train, y_test = train_test_split(drug_df.loc[:, drug_df.columns != 'Drug'], drug_df['Drug'])

# Uses different classifiers to train and test the dataset
def run_classifiers(file, X_train, X_test, y_train, y_test, stats):
    # Task 2.6a NB Classifier
    file.write(f'(a) ******************************** Gaussian NB Classifier ********************************\n')
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    file.write('\n(b) Confusion Matrix:\n')
    file.write(str(confusion_matrix(y_test, y_pred)))
    file.write('\n\n(c)(d) Classification Report:\n')
    file.write(str(classification_report(y_test, y_pred, zero_division=0)))

    # Task 2.8
    acc, macro, weighted = [], [], []
    for i in range(10):
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        acc.append(report['accuracy'])
        macro.append(report['macro avg']['f1-score'])
        weighted.append(report['weighted avg']['f1-score'])
    stats['gaussian_nb']['avg_accuracy'] = np.average(acc)
    stats['gaussian_nb']['avg_macro_F1'] = np.average(macro)
    stats['gaussian_nb']['avg_weighted_F1'] =  np.average(weighted)
    stats['gaussian_nb']['std_dev_accuracy'] = np.std(acc)
    stats['gaussian_nb']['std_dev_macro_F1'] = np.std(macro)
    stats['gaussian_nb']['std_dev_weighted_F1'] = np.std(weighted)

    # Task 2.6b Base-DT Classifier
    file.write(f'\n\n(a) ******************************** Base Decision Tree Classifier ********************************\n')
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    file.write('\n(b) Confusion Matrix:\n')
    file.write(str(confusion_matrix(y_test, y_pred)))
    file.write('\n\n(c)(d) Classification Report:\n')
    file.write(str(classification_report(y_test, y_pred, zero_division=0)))

    # Task 2.8
    acc, macro, weighted = [], [], []
    for i in range(10):
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        acc.append(report['accuracy'])
        macro.append(report['macro avg']['f1-score'])
        weighted.append(report['weighted avg']['f1-score'])
    stats['base_dt']['avg_accuracy'] = np.average(acc)
    stats['base_dt']['avg_macro_F1'] = np.average(macro)
    stats['base_dt']['avg_weighted_F1'] =  np.average(weighted)
    stats['base_dt']['std_dev_accuracy'] = np.std(acc)
    stats['base_dt']['std_dev_macro_F1'] = np.std(macro)
    stats['base_dt']['std_dev_weighted_F1'] = np.std(weighted)

    # Task 2.6c Top-DT Classifier
    file.write(f'\n\n(a) ******************************** Top Decision Tree Classifier ********************************\n')
    params = {'criterion': ['gini', 'entropy'], 'max_depth': [6, 17], 'min_samples_split': [5, 11, 19]}
    clf = GridSearchCV(DecisionTreeClassifier(), param_grid=params)
    clf.fit(X_train, y_train)
    file.write(f'Best hyper-parameters: {clf.best_params_}\n')
    y_pred = clf.predict(X_test)
    file.write('\n(b) Confusion Matrix:\n')
    file.write(str(confusion_matrix(y_test, y_pred)))
    file.write('\n\n(c)(d) Classification Report:\n')
    file.write(str(classification_report(y_test, y_pred, zero_division=0)))

    # Task 2.8
    acc, macro, weighted = [], [], []
    for i in range(10):
        clf = GridSearchCV(DecisionTreeClassifier(), param_grid=params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        acc.append(report['accuracy'])
        macro.append(report['macro avg']['f1-score'])
        weighted.append(report['weighted avg']['f1-score'])
    stats['top_dt']['avg_accuracy'] = np.average(acc)
    stats['top_dt']['avg_macro_F1'] = np.average(macro)
    stats['top_dt']['avg_weighted_F1'] =  np.average(weighted)
    stats['top_dt']['std_dev_accuracy'] = np.std(acc)
    stats['top_dt']['std_dev_macro_F1'] = np.std(macro)
    stats['top_dt']['std_dev_weighted_F1'] = np.std(weighted)

    # Task 2.6d Perceptron
    file.write(f'\n\n(a) ******************************** Perceptron Classifier ********************************\n')
    clf = Perceptron()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    file.write('\n(b) Confusion Matrix:\n')
    file.write(str(confusion_matrix(y_test, y_pred)))
    file.write('\n\n(c)(d) Classification Report:\n')
    file.write(str(classification_report(y_test, y_pred, zero_division=0)))

    # Task 2.8
    acc, macro, weighted = [], [], []
    for i in range(10):
        clf = Perceptron()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        acc.append(report['accuracy'])
        macro.append(report['macro avg']['f1-score'])
        weighted.append(report['weighted avg']['f1-score'])
    stats['perceptron']['avg_accuracy'] = np.average(acc)
    stats['perceptron']['avg_macro_F1'] = np.average(macro)
    stats['perceptron']['avg_weighted_F1'] =  np.average(weighted)
    stats['perceptron']['std_dev_accuracy'] = np.std(acc)
    stats['perceptron']['std_dev_macro_F1'] = np.std(macro)
    stats['perceptron']['std_dev_weighted_F1'] = np.std(weighted)

    # Task 2.6e Base-MLP
    file.write(f'\n\n(a) ******************************** Base Multi-Layered Perceptron Classifier ********************************\n')
    file.write('Modified hyper-parameters: hidden_layer_sizes=(100), activation=\'logistic\', solver=\'sgd\'\n')
    clf = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', solver='sgd')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    file.write('\n(b) Confusion Matrix:\n')
    file.write(str(confusion_matrix(y_test, y_pred)))
    file.write('\n\n(c)(d) Classification Report:\n')
    file.write(str(classification_report(y_test, y_pred, zero_division=0)))

    # Task 2.8
    acc, macro, weighted = [], [], []
    for i in range(10):
        clf = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', solver='sgd')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        acc.append(report['accuracy'])
        macro.append(report['macro avg']['f1-score'])
        weighted.append(report['weighted avg']['f1-score'])
    stats['base_mlp']['avg_accuracy'] = np.average(acc)
    stats['base_mlp']['avg_macro_F1'] = np.average(macro)
    stats['base_mlp']['avg_weighted_F1'] =  np.average(weighted)
    stats['base_mlp']['std_dev_accuracy'] = np.std(acc)
    stats['base_mlp']['std_dev_macro_F1'] = np.std(macro)
    stats['base_mlp']['std_dev_weighted_F1'] = np.std(weighted)

    # Task 2.6f Top-MLP
    file.write(f'\n\n(a) ******************************** Top Multi-Layered Perceptron Classifier ********************************\n')
    params = {'activation': ['logistic','identity','tanh','relu'], 'hidden_layer_sizes': [(30,50),(15,15,15)], 'solver': ['sgd','adam']}
    clf = GridSearchCV(MLPClassifier(), param_grid=params)
    clf.fit(X_train, y_train)
    file.write(f'Best hyper-parameters: {clf.best_params_}\n')
    y_pred = clf.predict(X_test)
    file.write('\n(b) Confusion Matrix:\n')
    file.write(str(confusion_matrix(y_test, y_pred)))
    file.write('\n\n(c)(d) Classification Report:\n')
    file.write(str(classification_report(y_test, y_pred, zero_division=0)))

    # Task 2.8
    acc, macro, weighted = [], [], []
    for i in range(10):
        clf = GridSearchCV(MLPClassifier(), param_grid=params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        acc.append(report['accuracy'])
        macro.append(report['macro avg']['f1-score'])
        weighted.append(report['weighted avg']['f1-score'])
    stats['top_mlp']['avg_accuracy'] = np.average(acc)
    stats['top_mlp']['avg_macro_F1'] = np.average(macro)
    stats['top_mlp']['avg_weighted_F1'] =  np.average(weighted)
    stats['top_mlp']['std_dev_accuracy'] = np.std(acc)
    stats['top_mlp']['std_dev_macro_F1'] = np.std(macro)
    stats['top_mlp']['std_dev_weighted_F1'] = np.std(weighted)

    file.write('\n******************************** Stats after running each model 10 times ********************************\n')
    for classifier in stats:
        file.write(f'\n{classifier}:\n\n')
        for key in stats[classifier]:
            file.write(f'{key}: {stats[classifier][key]}\n')
    print('Finished!')
    

stats = {'gaussian_nb': {'avg_accuracy': 0, 'avg_macro_F1': 0, 'avg_weighted_F1': 0,'std_dev_accuracy': 0, 'std_dev_macro_F1': 0, 'std_dev_weighted_F1': 0},
    'base_dt': {'avg_accuracy': 0, 'avg_macro_F1': 0, 'avg_weighted_F1': 0,'std_dev_accuracy': 0, 'std_dev_macro_F1': 0, 'std_dev_weighted_F1': 0},
    'top_dt': {'avg_accuracy': 0, 'avg_macro_F1': 0, 'avg_weighted_F1': 0,'std_dev_accuracy': 0, 'std_dev_macro_F1': 0, 'std_dev_weighted_F1': 0},
    'perceptron': {'avg_accuracy': 0, 'avg_macro_F1': 0, 'avg_weighted_F1': 0,'std_dev_accuracy': 0, 'std_dev_macro_F1': 0, 'std_dev_weighted_F1': 0},
    'base_mlp': {'avg_accuracy': 0, 'avg_macro_F1': 0, 'avg_weighted_F1': 0,'std_dev_accuracy': 0, 'std_dev_macro_F1': 0, 'std_dev_weighted_F1': 0},
    'top_mlp': {'avg_accuracy': 0, 'avg_macro_F1': 0, 'avg_weighted_F1': 0,'std_dev_accuracy': 0, 'std_dev_macro_F1': 0, 'std_dev_weighted_F1': 0}}

print('Running classifers and writing to file, this could take some time...')
with open('drugs-performance.txt', 'w') as file:
    run_classifiers(file, X_train, X_test, y_train, y_test, stats)
    