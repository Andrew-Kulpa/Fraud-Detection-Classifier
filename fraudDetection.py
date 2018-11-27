"""
    @author: Andrew Kulpa
    Based on the Kaggle Dataset for fraud detection using transaction data
"""
import tensorflow as tf
import numpy as np
import pandas as pd
tf.logging.set_verbosity(tf.logging.INFO)

# Generate training input from the csv file.
train_data = pd.read_csv("creditcard.csv", header='infer', usecols=["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"])
train_data = np.array(train_data).astype(np.float32) # train_data is (entries)x29

train_labels = pd.read_csv("wdbc_train.data", header='infer', usecols=["Class"]) # Read the fraud classifications
encoded_training_labels = np.array(pd.get_dummies(train_labels)) # one hot encode; train labels are of shape (entries)x2 ([1, 0] = '0', [0, 1] = '1' )

print(train_data[0])
print(encoded_training_labels[0])
