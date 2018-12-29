Fraudulent Transaction Classifier
============================
  * Google Document for Initial Paper: [AI Final Project - Fraudulent Transaction Classifier - Andrew Kulpa](https://docs.google.com/document/d/1dq9B8d_f-LKBcEQa_L3XMWFd_trr1gExbIuAs_qu8Oc/edit?usp=sharing)
  * Presentation: [Fraudulent Transaction Classification: Using an Artificial Neural Network](https://docs.google.com/presentation/d/1zW9uH7N2kFI3SHHz2nBdkB0_eU-1hMcOL0Fb4Ktur2Q/edit?usp=sharing)
## Problem

To adequately prevent fraudulent transactions from occurring, banks employ a 
number of methods to recognize when a fraudulent transaction happens as 
opposed to when a regular transactions occurs. The problem of recognizing 
these two groups of transactions is at its root a binary classification 
problem. Fraudulent transactions are relatively rare compared to normal 
transactions, though, so any given method used to classify these transactions 
also keep in mind the inherent bias of the data. As such, this project 
focuses on implementing an artificial neural network in Tensorflow using 
weighted classes and other methods to improve performance and accuracy in
classifying fraudulent credit card transactions.

## Data Set 
This research was conducted based on the actual transaction data 
hosted on Kaggle, an online community of data scientists and machine 
learners. The data, itself, originates from European credit card 
transaction data from September 2013. It contains just two days of 
transactions with a total of 284,807 entries. Of these transaction 
records, 0.172% are fraudulent. More specifically, 284,315 entries 
are not fraudulent while 492 are fraudulent . The input data is 
comprised of records with an elapsed time since the start of data 
collection, 28 features that were transformed using a principal 
component analysis (PCA) transformation, and the transaction amount. 
Each record also contains output data that is either a “0” for non 
fraudulent activity and a “1” for fraudulent activity.


## Methodology
While initial tests differed, the experiments were performed using a 
multi-layered neural network trained and tested upon data split at a 
60:40 ratio corresponding to the training and testing data. These 
were generated using Scikit’s “train_test_split” function and a defined 
seed to ensure reproduction of the included data. The 
resulting data contained 170587 non-fraudulent training entries, 
113728 non-fraudulent testing entries, 297 fraudulent training 
entries, and 195 fraudulent testing entries. The two classes of 
data were encoded using one-hot encoding.

Each model had a relatively high learning rate that gradually 
decayed, andlater models weighted the fraudulent transaction 
class much more to improverecall. The model is drawn below:

![ANN Architecture PNG](https://github.com/Andrew-Kulpa/Fraud-Detection-Classifier/blob/master/fraud-classifier-model.JPG)

### System Flow
The weights and biases between tensors were randomly generated 
in Tensorflow. The model was composed of an input layer of 29 
input nodes, 3 hidden layers with 50 nodes, 70 nodes, and 50 
nodes respectively, and an output layer of 2 nodes.

### Implementation
The initial hyperparameter configurations were as follows: a 
learning rate of 8%, a decay of 96%, and a decay at every 
100 steps. All weights and biases are initially created using 
automatically generated a Tensorflow random normal number 
generation method. The Adam optimization and Gradient 
Descent algorithms were both used to create some test models for 
comparison. Later models exclusively tuned the Adam optimized 
model due to its better performance.

## Final Results
This model achieved the highest accuracy for fraudulent recognition 
at a rate of at most 90.76%, while at that same point still retaining 
an accuracy of 99.05%. While this accuracy was high, this meant that 
1064 false positives existed at that point during testing. A much 
greater weight upon fraudulent transactions also resulted in a much 
more unstable model.

In reviewing the performance of the model, it became clear that as 
model became granularly accurate at classifying all transactions it 
lost fraudulent transaction recognition accuracy. This usually 
converted to above 99.05% accuracy. When it neared 99% accuracy 
it grew very accurate at recognizing frauds at a rate of at most 
90.76%, but as it neared 99.9% accuracy it would come close to 80% 
fraud recognition accuracy.

## Conclusion
Through many different iterations over the Adam and Gradient Descent 
based models, this project demonstrates that artificial neural networks 
are capable of learning to classify fraudulent transactions from extremely 
biased datasets. In addition, it was shown that hardware incompatibilities 
can greatly affect testing results. Experimental results show that the 
Tensorflow implementation of the Adam optimization algorithm converged upon 
higher accuracies for both testing and training datasets than the Gradient 
Descent optimizer. Also, class weights provide a relatively easy method to 
counteract dataset bias or adjust for class importance.

While increasing the size of each layer provides a relative increase in 
performance, the training process takes much longer than that of the model 
in the final iteration. For this reason, future work may first be based upon 
trying out a much larger neural network with more nodes per layer or more layers. 
Furthermore, a model utilizing rectified linear units or a convolutional neural 
network could possibly improve performance further. Given that either of these 
proposed models could require more neurons and layers resulting in overfitting, 
the Dropout algorithm could be applied as well.








