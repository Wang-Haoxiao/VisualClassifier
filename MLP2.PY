import numpy as np
from scipy.special import expit
import sys

class NeuralNetMLP2(object):
    def __init__(self, n_output, n_features, hidden_layers=[30], l1=0.0, l2=0.0, epochs=500, eta=0.001,
                 alpha=0.0, decrease_const=0.0, shuffle=True, minibatches=1, random_state=None):
        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.hidden_layers = hidden_layers
        self.n_layers = 1 + len(hidden_layers)  # Plus one for output layer
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
        """Initialize weights with small random numbers."""
        layers = [self.n_features] + self.hidden_layers + [self.n_output]
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.uniform(-1.0, 1.0, size=layers[i + 1] * (layers[i] + 1))
            w = w.reshape(layers[i + 1], layers[i] + 1)
            weights.append(w)
        return weights

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)

        Uses scipy.special.expit to avoid overflow
        error for very small input values z.

        """
        # return 1.0 / (1.0 + np.exp(-z))
        return expit(z)

    def _add_bias_unit(self, X, how='column'):
        """Add bias unit (column or row of 1s) to array at index 0.

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            Input array.
        how : str (default: 'column')
            Defines how the bias unit is added -- as a column or as a row.

        Returns
        -------
        X_new : array, shape = [n_samples, n_features + 1] or [n_samples + 1, n_features]
            Input array with added bias unit.
        """
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError("`how` must be `column` or `row`")
        return X_new

    # ... [rest of the methods remain unchanged, except for _feedforward, _get_cost, _get_gradient, and predict]
    def _encode_labels(self, y, k):
        """Encode labels into one-hot representation

        Parameters
        -----------
        y : array, shape = [n_samples]
            Target class labels.
        k : int
            Number of unique class labels (number of columns in the one-hot representation).

        Returns
        -------
        onehot : array, shape = (k, n_samples)
        """
        onehot = np.zeros((k, y.shape[0]))
        for idx, label in enumerate(y):
            onehot[label, idx] = 1.0
        return onehot



    def _feedforward(self, X, weights):
        a = [X]
        for i, w in enumerate(weights):
            a[i] = self._add_bias_unit(a[i], how='column')
            net_input = w.dot(a[i].T)
            out = self._sigmoid(net_input)
            a.append(out.T)
        return a  # Contains all activations per layer

    # Modify _get_cost to handle multiple layers
    def _get_cost(self, y_enc, output, weights):
        """Compute the cost function.

        Parameters:
        y_enc : array, shape = (n_labels, n_samples)
            one-hot encoded class labels.
        output : array, shape = [n_output_units, n_samples]
            Activation of the output layer.
        weights : list of arrays
            Weight matrices for each layer.

        Returns:
        cost : float
            Regularized cost.
        """
        # print(output.T.shape)
        term1 = -y_enc * np.log(output.T)
        term2 = (1.0 - y_enc) * np.log(1.0 - output.T)
        cost = np.sum(term1 - term2)
        L1_term = sum([np.abs(w[:, 1:]).sum() for w in weights])
        L2_term = sum([(w[:, 1:] ** 2).sum() for w in weights])
        cost = cost + self.l1 * L1_term / 2.0 + self.l2 * L2_term / 2.0
        return cost
    
    # Modify _get_gradient for backpropagation through all layers
    def _get_gradient(self, a, y_enc, weights):
        """ Compute the gradient using backpropagation.

        Parameters:
        a : list of arrays
            Activation values for all layers
        y_enc : array, shape = (n_labels, n_samples)
            One-hot encoded class labels.
        weights : list of arrays
            Weight matrices for each layer.

        Returns:
        grad : list of arrays
            Gradient of the weight matrices for each layer.
        """
        delta = a[-1].T - y_enc
        grad = [delta.dot(a[-2])]

        for i in range(len(weights) - 2, 0, -1):
            delta = weights[i].T.dot(delta) * self._sigmoid_gradient(a[i])
            delta = delta[1:, :]  # Skip bias unit
            grad.insert(0, delta.dot(a[i-1].T))

        # Regularize
        for i in range(len(grad)):
            # print(f"Layer {i}, grad shape: {grad[i].shape}, weights shape: {weights[i+1].shape}")
            # print(f"Layer {i}, grad shape: {grad[i].shape}, weights shape: {weights[i-1].shape}")
            grad[i][:, 1:] += self.l2 * weights[i+1][:, 1:]
            grad[i][:, 1:] += self.l1 * np.sign(weights[i+1][:, 1:])
        return grad

    def fit(self, X, y, print_progress=False):
        """ Learn weights from training data.

        Parameters:
        X : array, shape = [n_samples, n_features]
            Input layer with original features.
        y : array, shape = [n_samples]
            Target class labels.
        print_progress : bool (default: False)
            Prints progress to stderr.

        Returns:
        self
        """
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_prev = [np.zeros(w.shape) for w in self.weights]

        for i in range(self.epochs):
            # Adaptive learning rate
            self.eta /= (1 + self.decrease_const * i)

            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i + 1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_enc = X_data[idx], y_enc[:, idx]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                # feedforward
                a = self._feedforward(X_data[idx], self.weights)

                cost = self._get_cost(y_enc=y_enc[:, idx], output=a[-1], weights=self.weights)
                self.cost_.append(cost)

                # compute gradient via backpropagation
                grad = self._get_gradient(a, y_enc[:, idx], self.weights)

                for j in range(0, len(self.weights) - 1):
                    delta = self.eta * grad[j]
                    # print("weights", self.weights[j+1].shape)
                    # print(delta.shape,  delta_prev[j].shape)
                    self.weights[j+1] -= (delta + (self.alpha * delta_prev[j+1]))
                    delta_prev[j+1] = delta

        return self

    def predict(self, X):
        """Predict class labels"""
        # Modify to handle multiple layers
        a = self._feedforward(X, self.weights)
        y_pred = np.argmax(a[-1], axis=1)
        return y_pred

    # ... [fit and partial_fit methods need slight modifications to work with new architecture]

# Continue with the implementation of the remaining methods like _get_cost, _get_gradient, etc.
