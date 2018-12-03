"""
	@author: Andrew Kulpa
	Based on the Kaggle Dataset for fraud detection using transaction data
"""
import tensorflow as tf
import numpy as np
import pandas as pd
tf.logging.set_verbosity(tf.logging.INFO)

########################################################
######                Read in Files               ######
########################################################
print("Reading in files...")
train_data_csv = 'creditcard_train.csv'
test_data_csv = 'creditcard_test.csv'

train_data = pd.read_csv(train_data_csv) # DataFrame object
trainX = train_data.drop(['Class'], axis='columns')
trainY = train_data.Class
train_encoded_labels = np.array(pd.get_dummies(trainY)) 

test_data = pd.read_csv(test_data_csv) # DataFrame object
testX = test_data.drop(['Class'], axis='columns')
testY = test_data.Class
test_encoded_labels = np.array(pd.get_dummies(testY)) 

########################################################
######                 Define Model               ######
########################################################
print("Defining the model...")
# Parameters
useAdamOptimizer = False
initial_learning_rate = 0.08
training_epochs = 1001
decay_steps = 50
display_epoch = 20
decay_base_rate = 0.96

# Network
input_nodes = 29
hidden_1_nodes = 50
hidden_2_nodes = 70
hidden_3_nodes = 50
classes = 2

# Create placeholder tensors
Input = tf.placeholder("float", [None, input_nodes])
Output = tf.placeholder("float", [None, classes]) # expecting [1, 0] or [0, 1]

# Create hashes of weights and biases, shaped to conform to the inputs and preceding layers
weights = {
	'hidden1': tf.Variable(tf.random_normal([input_nodes, hidden_1_nodes])),
	'hidden2': tf.Variable(tf.random_normal([hidden_1_nodes, hidden_2_nodes])),
	'hidden3': tf.Variable(tf.random_normal([hidden_2_nodes, hidden_3_nodes])),
	'output': tf.Variable(tf.random_normal([hidden_3_nodes, classes])),
}
biases = {
	'bias1': tf.Variable(tf.random_normal([hidden_1_nodes])),
	'bias2': tf.Variable(tf.random_normal([hidden_2_nodes])),
	'bias3': tf.Variable(tf.random_normal([hidden_3_nodes])),
	'output': tf.Variable(tf.random_normal([classes])),
}

# Create the model for the multi-layered neural network, utilizing the previous weights and biases.
def mlnn(input_x):
	hidden1 = tf.add(tf.matmul(input_x, weights['hidden1']), biases['bias1'])
	hidden2 = tf.add(tf.matmul(hidden1, weights['hidden2']), biases['bias2'])
	hidden3 = tf.add(tf.matmul(hidden2, weights['hidden3']), biases['bias3'])
	logits = tf.add(tf.matmul(hidden3, weights['output']), biases['output'])
	return logits

# Define the model, loss, optimizer, and learning rate
logits = mlnn(Input)

global_step = tf.Variable(0, trainable=False) # https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_base_rate, staircase=True)
if(useAdamOptimizer):
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Output))
	train_op = optimizer.minimize(loss_op, global_step = global_step)
else:
	learning_rate *= 0.001 # hopefully prevent exploding costs
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	print("The gradient descent optimizer is currently working poorly.. if not at all. Just use the adam optimization method for now.")
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Output))
	grad_vars = optimizer.compute_gradients(loss_op)
	grad = [x[0] for x in grad_vars]
	variables = [x[1] for x in grad_vars]
	train_op = optimizer.apply_gradients(zip(grad, variables), global_step=global_step)

print("Initializing variables and data collection arrays...")
# initialize data for plotting accuracy + cost
training_accuracy_data = []
testing_accuracy_data = []
cost_data = []

# Initializing the variables
init = tf.global_variables_initializer()
false_positives = 0
false_negatives = 0
true_positives = 0
true_negatives = 0

with tf.Session() as sess:
	print("Beginning training of the model...")
	sess.run(init)

	# training
	for epoch in range(training_epochs):
		# Backpropagation optimization and cost operation
		_, cost = sess.run([train_op, loss_op], feed_dict={Input: trainX, Output: train_encoded_labels})
		print(cost)
		# Display results for each epoch cycle
		if epoch % display_epoch == 0:
			print("Epoch:", '%04d' % (epoch+1))
			pred = tf.nn.softmax(logits)
			fp = tf.metrics.false_positives(tf.argmax(Output, 1),tf.argmax(pred, 1))
			tp = tf.metrics.true_positives(tf.argmax(Output, 1),tf.argmax(pred, 1))
			fn = tf.metrics.false_negatives(tf.argmax(Output, 1),tf.argmax(pred, 1))
			tn = tf.metrics.true_negatives(tf.argmax(Output, 1),tf.argmax(pred, 1))
			# acc = tf.metrics.mean_per_class_accuracy(tf.argmax(Output, 1),tf.argmax(pred, 1), num_classes=2)
			sess.run(tf.local_variables_initializer())
			metrics = sess.run([fp, tp, fn, tn], feed_dict={Input: testX, Output: test_encoded_labels})
			print("\tTesting Confusion Metrics:")
			print('\t\ttrue_negative: %d' % (metrics[3][0]))
			print('\t\tfalse_positive: %d' % (metrics[0][0]))
			print('\t\ttrue_positive: %d' % (metrics[1][0]))
			print('\t\tfalse_negative: %d' % (metrics[2][0]))

			lr = sess.run(optimizer._lr) if useAdamOptimizer else sess.run(optimizer._learning_rate)
			print("\ttraining_cost={:.9f}".format(cost), "learning_rate={:.5f}".format(lr))
			# Apply softmax to logits
			pred = tf.nn.softmax(logits)
			correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Output, 1))
			# Calculate the testing_accuracy of the model
			testing_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			testing_accuracy_amt = testing_accuracy.eval({Input: testX, Output: test_encoded_labels})
			print("\tTest Data Accuracy:", testing_accuracy_amt)
			testing_accuracy_data.append([epoch, testing_accuracy_amt])
			
			# Calculate the training_accuracy of the model
			pred1 = tf.nn.softmax(logits)
			correct_prediction1 = tf.equal(tf.argmax(pred1, 1), tf.argmax(Output, 1))
			training_accuracy = tf.reduce_mean(tf.cast(correct_prediction1, "float"))
			training_accuracy_amt = training_accuracy.eval({Input: trainX, Output: train_encoded_labels})
			print("\tTraining Data Accuracy:", training_accuracy_amt, "\n")
			training_accuracy_data.append([epoch, training_accuracy_amt])

			# including cost data
			cost_data.append([epoch, cost])

	print("Training done!")
	print("Generating performance graphs...")
	# plot and save pdf for the testing accuracy data
	training_accuracy_df = pd.DataFrame(data=training_accuracy_data,columns=['epoch','accuracy'])
	training_accuracy_figure = training_accuracy_df.plot(x='epoch', y='accuracy', kind='line', title='Memorization Accuracy').get_figure()
	training_accuracy_figure.savefig('training_accuracy_graph.pdf')

	# plot and save pdf for the training accuracy data
	testing_accuracy_df = pd.DataFrame(data=testing_accuracy_data,columns=['epoch','accuracy'])
	testing_accuracy_figure = testing_accuracy_df.plot(x='epoch', y='accuracy', kind='line', title='Generalization Accuracy').get_figure()
	testing_accuracy_figure.savefig('testing_accuracy_graph.pdf')

	# plot and save pdf for the cost data
	cost_df = pd.DataFrame(data=cost_data,columns=['epoch','cost'])
	cost_figure = cost_df.plot(x='epoch', y='cost', kind='line', title='Memorization Deficiency').get_figure()
	cost_figure.savefig('cost_graph.pdf')