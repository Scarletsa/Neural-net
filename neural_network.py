import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/data/", one_hot=True)

# setting the different number of nodes in each layer of the network.
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10 # The number of things we're trying to classify. In our case the 10 digits.
batch_size = 100 # How many inputs we want to process at a time so we don't overflow RAM.

x = tf.placeholder('float', [None, 784]) # Second position in this tensor must be the same as the first position in layer 1 tensor due to matrix multiplication.
y = tf.placeholder('float')

def neural_network_model(data):

    # Since we don't have any assumptions for the pixels in the images at first, we're going to start with random weights and biases.
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}

    # Feeding the data through the model.
    # (weights * data) + biases
    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']),  hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1) # Activation function to initialize tensor.

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']),  hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']),  hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output # returning the "one hot" array of what we think the number is.

def train_neural_network(x):
    prediction = neural_network_model(x) # Gives us a prediction "one hot" array to compare against the known label.
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y)) # Averages out (reduce_mean) the probability error in matching predictions to classes (softmax_cross_entropy_with_logits_v2) across tensor.
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam').minimize(cost) # Used during backpropogation, changing the weights to help minimize the cost, or probability error.

    epochs = 10 # Number of learning cycles.

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # starts a training session

        # feed forward + backpropagation = epoch, or one cycle
        for epoch in range(epochs):
            epoch_loss = 0

            # Neural networks are trained with incredibly large datasets, so we process them in batches. You likely wouldn't be able to hold all the data in memory at once.
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size) # Feed forward
                # print(epoch_x) # Input data after weights and biases
                # print(epoch_y) # One hot label to tell the machine what number it actually is.
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y}) # Backpropagation: the machine going backwards to adjust the weights based on how wrong it was in it's prediction.
                # print(c) # Showing how wrong the machine was in it's prediction.
                epoch_loss += c # summation of the errors made for examples in training sets in the cycle
                # input() # Interupt execution to see individual predictions.

            print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss) # Showing how inaccurate the machine was in the epoch.

        correct_predictions = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))  # Argmax returns the index of the highest value. We're then comparing equality of each index position in the tensors.
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float')) # Changing the boolean tensor back into the float variable weights, and taking an average of the correct predictions.
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels})) # Testing our machine against a different dataset to check accuracy.

train_neural_network(x) # init
