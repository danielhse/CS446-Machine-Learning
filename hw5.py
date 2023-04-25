import hw5_utils as utils
import numpy as np
import torch
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    n = x_train.shape[0]
    alpha = torch.zeros(n, requires_grad=True)

    for _ in range(num_iters):
        # Calculate the objective function
        obj = 0.5 * sum(alpha[i] * alpha[j] * y_train[i] * y_train[j] * kernel(x_train[i], x_train[j])
                       for i in range(n) for j in range(n)) - alpha.sum()
        
        # Compute gradients
        obj.backward()

        # Update alpha using gradient descent
        with torch.no_grad():
            alpha -= lr * alpha.grad
            alpha.grad.zero_()

            # Project alpha back onto the feasible set
            if c is None:
                alpha.clamp_(min=0)
            else:
                alpha.clamp_(min=0, max=c)

            # Re-enable gradient calculation for alpha
            alpha.requires_grad_(True)

    return alpha.detach()

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''
    m = x_test.shape[0]
    predictions = torch.empty(m)

    for i in range(m):
        pred = sum(alpha[j] * y_train[j] * kernel(x_train[j], x_test[i]) for j in range(alpha.shape[0]))
        predictions[i] = pred

    return predictions

# Load XOR data
x_train, y_train = utils.xor_data()

# Train the SVM with the RBF kernel with σ = 1
alpha = svm_solver(x_train, y_train, lr=0.1, num_iters=10000, kernel=utils.rbf(sigma=4))

# Define the predictor function
def predictor(x_test):
    return svm_predictor(alpha, x_train, y_train, x_test, kernel=utils.rbf(sigma=4))

# Plot the contour lines
utils.svm_contour(predictor, xmin=-5, xmax=5, ymin=-5, ymax=5)


def logistic(X, Y, lrate=.01, num_iter=10000000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    n, d = X.shape
    X = torch.cat([torch.ones(n, 1), X], dim=1)  # Prepend a column of ones to X
    w = torch.zeros(d + 1, 1)  # Initialize w with zeros

    for _ in range(num_iter):
        z = torch.matmul(X, w) * Y
        gradient = torch.mean(-Y * X * torch.exp(-z) / (1 + torch.exp(-z)), dim=0, keepdim=True).T
        w = w - lrate * gradient

    return w

def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    X, Y = utils.load_logistic_data()
    w_logistic = logistic(X, Y)
    w_ols = utils.linear_normal(X, Y)

    # Plot the data
    plt.scatter(X[:, 0], X[:, 1], label='Dataset')

    # Plot decision boundary for logistic regression
    x = torch.linspace(-5, 5, 1000)
    y = (-w_logistic[0] - w_logistic[1] * x) / w_logistic[2]
    plt.plot(x, y, label='Logistic Regression', color='red')

    # Plot decision boundary for least squares
    y_ols = (-w_ols[0] - w_ols[1] * x) / w_ols[2]
    plt.plot(x, y_ols, label='Least Squares', color='green')

    plt.title('Logistic vs OLS Plot')
    plt.xlabel('X_1')
    plt.ylabel('X_2')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    logistic_vs_ols()
