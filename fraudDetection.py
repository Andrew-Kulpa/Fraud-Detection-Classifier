"""
	@author: Andrew Kulpa
	Based on the Kaggle Dataset for fraud detection using transaction data
"""
import tensorflow as tf
import numpy as np
import pandas as pd
tf.logging.set_verbosity(tf.logging.INFO)
#### https://github.com/openAGI/tefla/blob/master/tefla/core/base.py
####

def clip_grad_norms(self, gradients_to_variables, max_norm=5):
	"""Clips the gradients by the given value.
	Args:
		gradients_to_variables: A list of gradient to variable pairs (tuples).
		max_norm: the maximum norm value.
	Returns:
		A list of clipped gradient to variable pairs.
	"""
	grads_and_vars = []
	for grad, var in gradients_to_variables:
		if grad is not None:
			if isinstance(grad, tf.IndexedSlices):
				tmp = tf.clip_by_norm(grad.values, max_norm)
				grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
			else:
				grad = tf.clip_by_norm(grad, max_norm)
		grads_and_vars.append((grad, var))
	return grads_and_vars

########################################################
######                Read in Files               ######
########################################################
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

# Parameters
useAdamOptimizer = True
initial_learning_rate = 0.08
training_epochs = 10000
decay_steps = 100
display_epoch = 50
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
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	print("The gradient descent optimizer is currently working poorly.. if not at all. Just use the adam optimization method for now.")
	max_grad_norm = 100
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Output))
	grad_vars = optimizer.compute_gradients(loss_op)
	grad = [x[0] for x in grad_vars]
	variables = [x[1] for x in grad_vars]
	grad, grad_norm = tf.clip_by_global_norm(grad, max_grad_norm)
	train_op = optimizer.apply_gradients(zip(grad, variables), global_step=global_step)


# initialize data for plotting accuracy + cost
cost_data = []
accuracy_data = []

# Initializing the variables
init = tf.global_variables_initializer()
false_positives = 0
false_negatives = 0
true_positives = 0
true_negatives = 0
with tf.Session() as sess:
	sess.run(init)
	
	# print(tf.equal(tf.argmax(pred, 1), tf.argmax(Output, 1)).eval())# true_negatives = tf.equal(tf.equal(tf.argmax(pred, 1), tf.argmax(Output, 1)),tf.constant(0))
	#training
	for epoch in range(training_epochs):
		# Backpropagation optimization and cost operation
		_, cost = sess.run([train_op, loss_op], feed_dict={Input: trainX, Output: train_encoded_labels})
		# Display results for each epoch cycle
		if epoch % display_epoch == 0:
			pred = tf.nn.softmax(logits)
			acc = tf.metrics.mean_per_class_accuracy(tf.argmax(Output, 1),tf.argmax(pred, 1), num_classes=2)
			sess.run(tf.local_variables_initializer())
			acc_matrix = sess.run(acc, feed_dict={Input: testX, Output: test_encoded_labels})
			true_negative = acc_matrix[1].tolist()[0][0]
			false_positive = acc_matrix[1].tolist()[0][1]
			true_positive = acc_matrix[1].tolist()[1][0]
			false_negative = acc_matrix[1].tolist()[1][1]
			print('true_negative: %d' % (true_negative))
			print('false_positive: %d' % (false_positive))
			print('true_positive: %d' % (true_positive))
			print('false_negative: %d' % (false_negative))

			lr = sess.run(optimizer._lr) if useAdamOptimizer else sess.run(optimizer._learning_rate)
			print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(cost), "learning_rate={:.5f}".format(lr))
			# Apply softmax to logits
			pred = tf.nn.softmax(logits)
			correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Output, 1))
			# Calculate the accuracy of the model
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			accuracy_amt = accuracy.eval({Input: testX, Output: test_encoded_labels})
			print("Test Data Accuracy:", accuracy_amt, "\n")
			cost_data.append([epoch, cost])
			accuracy_data.append([epoch, accuracy_amt])
	print("Training done!")
	# plot and save pdf for the data
	cost_df = pd.DataFrame(data=cost_data,columns=['epoch','cost'])
	cost_figure = cost_df.plot(x='epoch', y='cost', kind='line', title='Memorization Deficiency').get_figure()
	cost_figure.savefig('cost_graph.pdf')

	accuracy_df = pd.DataFrame(data=accuracy_data,columns=['epoch','accuracy'])
	accuracy_figure = accuracy_df.plot(x='epoch', y='accuracy', kind='line', title='Generalization Accuracy').get_figure()
	accuracy_figure.savefig('accuracy_graph.pdf')