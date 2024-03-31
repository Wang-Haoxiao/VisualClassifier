import numpy as np
from scipy.special import expit
import sys


# Implémentation du MLP
# Couche d'entrées, 1 couche cachée et couche de sorties
#
# Code similaire à Adaline

class NeuralNetMLP2(object):
    def __init__(self, layers, l1=0.0, l2=0.0, epochs=500, eta=0.001, alpha=0.0, decrease_const=0.0, shuffle=True, minibatches=1, random_state=None):
        np.random.seed(random_state)
        self.layers = layers
        self.n_layers = len(layers)
        self.weights = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _initialize_weights(self):
        weights = []
        for i in range(1, len(self.layers)):
            w = np.random.uniform(-1.0, 1.0, size=self.layers[i] * (self.layers[i-1] + 1))
            w = w.reshape(self.layers[i], self.layers[i-1] + 1)
            weights.append(w)
        return weights

    def _encode_labels(self, y, n_classes):
        """Encode labels into one-hot representation

        Parameters:
        -----------
        y : array, shape = [n_samples]
            Target values.
        n_classes : int
            Number of classes

        Returns:
        --------
        onehot : array, shape = (n_labels, n_samples)
        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _add_bias_unit(self, X, how='column'):
        """Add bias unit (column or row of 1s) to array at index 0"""
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new


    # Activation Functions and Their Gradients
    def _activation(self, z):
        # Sigmoid function can be used for hidden layers
        return expit(z)

    def _activation_gradient(self, z):
        # Gradient of the sigmoid function
        a = self._activation(z)
        return a * (1 - a)

    def _output_activation(self, z):
        # For the output layer, you can use softmax or sigmoid, depending on your task
        # Here's sigmoid for binary classification
        return expit(z)

    # Cost Function
    def _get_cost(self, y_enc, output):
        """Compute the cost function."""
        term1 = -y_enc * np.log(output)
        term2 = (1 - y_enc) * np.log(1 - output)
        cost = np.sum(term1 - term2)
        # Add regularization term if l1 or l2 are not zero
        L1_term, L2_term = 0, 0
        for w in self.weights:
            L1_term += np.abs(w[:, 1:]).sum()
            L2_term += (w[:, 1:] ** 2).sum()
        cost += (self.l1 / 2.0) * L1_term + (self.l2 / 2.0) * L2_term
        return cost

    # update weights 
    def _update_weights(self, grad_w, delta_w_prev):
        new_weights = []
        new_delta_w = []
        for w, grad, delta_w in zip(self.weights, grad_w, delta_w_prev):
            new_delta = self.eta * grad + self.alpha * delta_w
            new_weights.append(w - new_delta)
            new_delta_w.append(new_delta)
        return new_weights, new_delta_w

    # predict function 
    def predict(self, X):
        """Predict class labels."""
        # Feedforward
        activations, _ = self._feedforward(X)
        output = activations[-1]
        # Convert output activations to class labels
        y_pred = np.argmax(output, axis=0)
        return y_pred


    def _feedforward(self, X):
        print("X=", X)
        activation = X
        activations = [X]  # List to store all activations, layer by layer
        zs = []  # List to store all z vectors, layer by layer

        for i in range(len(self.weights)):
            if i > 0:  # Add bias unit to the activation of the previous layer (except for the input layer)
                activation = np.hstack((np.ones((activation.shape[0], 1)), activation))
            z = np.dot(activation, self.weights[i].T)  # Dot product
            activation = expit(z) if i < len(self.weights) - 1 else z  # Sigmoid for hidden layers, linear for output
            activations.append(activation)  # Store activations
            zs.append(z)  # Store z vectors

        return activations, zs

    def _backprop(self, activations, zs, y_enc):
        grad_w = [np.zeros(w.shape) for w in self.weights]
        # Output error
        delta = activations[-1] - y_enc
        grad_w[-1] = delta.dot(activations[-2].T)

        # Backpropagate the error
        for l in range(2, self.n_layers):
            z = zs[-l]
            delta = self.weights[-l+1].T.dot(delta) * self._activation_gradient(z)
            delta = delta[1:, :]  # Skip the bias term
            grad_w[-l] = delta.dot(activations[-l-1].T)

        return grad_w

    # update fit method
    def fit(self, X, y, print_progress=False):
        """ Learn weights from training data.

        Parameters:
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.
        y : array, shape = [n_samples]
            Target class labels.
        print_progress : bool (default: False)
            Prints progress as the number of epochs to stderr.

        Returns:
        --------
        self
        """
        self.cost_ = []
        y_enc = self._encode_labels(y, self.layers[-1])  # one-hot encoding

        # Initialize previous gradient for momentum-based update
        delta_w_prev = [np.zeros(w.shape) for w in self.weights]

        for i in range(self.epochs):
            # Adaptive learning rate
            self.eta /= (1 + self.decrease_const * i)

            if self.shuffle:
                indices = np.random.permutation(y.shape[0])
                X, y_enc = X[indices], y_enc[:, indices]

            mini_batches = np.array_split(range(y.shape[0]), self.minibatches)
            for idx in mini_batches:
                # Feedforward
                print("idx=", idx)
                activations, zs = self._feedforward(X[idx])

                # Compute cost
                cost = self._get_cost(y_enc=y_enc[:, idx], output=activations[-1])
                self.cost_.append(cost)

                # Backpropagation
                grad_w = self._backprop(activations, zs, y_enc[:, idx])

                # Update weights
                self.weights, delta_w_prev = self._update_weights(grad_w, delta_w_prev)

            if print_progress:
                print('Epoch: {}/{} | Cost: {:.4f}'.format(i+1, self.epochs, cost))

        return self

