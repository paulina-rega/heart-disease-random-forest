import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel


def one_hot_encode(df, columns_to_encode):
    for col in columns_to_encode:
        one_hot = pd.get_dummies(df[col], prefix=col)
        df = df.drop(col, axis=1)
        df = df.join(one_hot)
    return df


df = pd.read_csv('Heart-Disease-UCI.csv')
X = df.drop(['target'], axis = 1)
Y = df['target'] # 0 = disease, 1 = no disease


# encoding non-numeric values
columns_to_encode = ['thal', 'sex', 'cp', 'restecg', 'slope']
X = one_hot_encode(X, columns_to_encode)

# training model
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

clf = RandomForestClassifier(n_estimators=100, random_state = 1, n_jobs =-1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print('Accuracy (all given parameters considered): {}'.format(
    metrics.accuracy_score(y_test, y_pred)))

 
labels = list(X.columns)
'''
# Uncomment this code block to show all features importance

print('\nFeatures importance: ')
for feature in zip(labels, clf.feature_importances_):
    print(\nfeature)
'''

# choosing important fatures for new model
sfm = SelectFromModel(clf, threshold = 0.03)
sfm.fit(x_train, y_train)

# trainning model with only important features
x_important_train = sfm.transform(x_train)
x_important_test = sfm.transform(x_test)
clf_important = RandomForestClassifier(n_estimators=100, random_state = 1, 
                                       n_jobs =-1)
clf_important.fit(x_important_train, y_train)
y_important_pred = clf_important.predict(x_important_test)


print('\nMost important features:')

for feature_list_index in sfm.get_support(indices=True):
    print('\t - ' + labels[feature_list_index])

print('\nAccuracy (chosen parameters with highest importance): {}'.format(
    metrics.accuracy_score(y_important_pred, y_pred)))

