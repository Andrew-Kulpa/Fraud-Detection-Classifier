# https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
# https://medium.com/themlblog/splitting-csv-into-train-and-test-data-1407a063dd74
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Generate training input from the csv file.
data_source = 'creditcard.csv'
data = pd.read_csv(data_source) # DataFrame object
print(data.head())
class_count = data.Class.value_counts()
print('Total Entries:', class_count[0]+class_count[1])
print('Class 0:', class_count[0])
print('Class 1:', class_count[1])
print('Proportion:', round(class_count[0] / class_count[1], 2), ': 1')
figure = class_count.plot(kind='bar', title='Count (Class)').get_figure()
figure.savefig('class_histogram.pdf')

X = data.drop(['Class', 'Time'], axis='columns')
y = data.Class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=3)

train_set = pd.concat([X_train, y_train], axis=1)
train_set.to_csv('creditcard_train.csv', encoding='utf-8', index=False)
test_set = pd.concat([X_test, y_test], axis=1)
test_set.to_csv('creditcard_test.csv', encoding='utf-8', index=False)