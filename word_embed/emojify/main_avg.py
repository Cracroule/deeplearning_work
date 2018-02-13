import numpy as np
import pandas as pd
from emo_utils import read_csv, label_to_emoji, convert_to_one_hot, read_glove_vecs, predict, softmax, print_predictions,\
    plot_confusion_matrix
import emoji
import matplotlib.pyplot as plt


def main():
    X_train, Y_train = read_csv('../data/train_emoji.csv')
    X_test, Y_test = read_csv('../data/tesss.csv')
    maxLen = len(max(X_train, key=len).split())

    for index in range(10):
        print(X_train[index], label_to_emoji(Y_train[index]))

    Y_oh_train = convert_to_one_hot(Y_train, C=5)
    Y_oh_test = convert_to_one_hot(Y_test, C=5)

    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('../data/glove.6B.50d.txt')

    word = "cucumber"
    index = 289846
    print("")
    print("the index of", word, "in the vocabulary is", word_to_index[word])
    print("the", str(index) + "th word in the vocabulary is", index_to_word[index])

    pred, W, b = model(X_train, Y_train, word_to_vec_map)

    print("Training set:")
    pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
    print('Test set:')
    pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)

    X_my_sentences = np.array(
        ["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "not feeling happy"])
    Y_my_labels = np.array([[0], [0], [2], [1], [4], [3]])
    print('--------- PRINT PREDICTIONS ----------')
    pred = predict(X_my_sentences, Y_my_labels, W, b, word_to_vec_map)
    print_predictions(X_my_sentences, pred)

    print(Y_test.shape)
    print('           ' + label_to_emoji(0) + '    ' + label_to_emoji(1) + '    ' + label_to_emoji(
        2) + '    ' + label_to_emoji(3) + '   ' + label_to_emoji(4))
    print(pd.crosstab(Y_test, pred_test.reshape(56, ), rownames=['Actual'], colnames=['Predicted'], margins=True))
    plot_confusion_matrix(Y_test, pred_test)
    plt.show()


# GRADED FUNCTION: sentence_to_avg
def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.

    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation

    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
    """

    ### START CODE HERE ###
    # Step 1: Split sentence into list of lower case words (≈ 1 line)
    words = sentence.lower().split()

    # Initialize the average word vector, should have the same shape as your word vectors.
    avg = np.zeros((50,))

    # Step 2: average the word vectors. You can loop over the words in the list "words".
    for w in words:
        avg += word_to_vec_map[w]
    avg = avg / len(words)

    # the below line looks better to me than all previous ones
    # avg = np.average([word_to_vec_map[w] for w in words], axis=0)

    ### END CODE HERE ###
    return avg


# GRADED FUNCTION: model
def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
    """
    Model to train word vector representations in numpy.

    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m, 1)
    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations

    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """

    np.random.seed(1)

    # Define number of training examples
    m = Y.shape[0]  # number of training examples
    n_y = 5  # number of classes
    n_h = 50  # dimensions of the GloVe vectors

    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    # Convert Y to Y_onehot with n_y classes
    Y_oh = convert_to_one_hot(Y, C=n_y)

    # Optimization loop
    for t in range(num_iterations):  # Loop over the number of iterations
        for i in range(m):  # Loop over the training examples

            ### START CODE HERE ### (≈ 4 lines of code)
            # Average the word vectors of the words from the i'th training example
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # Forward propagate the avg through the softmax layer
            z = np.dot(W, avg) + b
            a = softmax(z)

            # Compute cost using the i'th training label's one hot representation and "A" (the output of the softmax)
            cost = - np.sum(Y_oh[i] * np.log(a))
            ### END CODE HERE ###

            # print('---')
            # print(a)
            # print(Y_oh.shape)
            # print(cost)

            # Compute gradients
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
            db = dz

            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db

        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map)

    return pred, W, b

if __name__ == '__main__':
    main()

