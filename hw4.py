import torch
import hw4_utils as utils
import matplotlib.pyplot as plt

'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

    Be sure to modify your input matrix X in exactly the way specified. That is,
    make sure to prepend the column of ones to X and not put the column anywhere
    else, and make sure your feature-expanded matrix in Problem 3 is in the
    specified order (otherwise, your w will be ordered differently than the
    reference solution's in the autograder).
'''

# Problem 2
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    n, d = X.shape
    X = torch.cat([torch.ones(n, 1), X], dim=1)  # Prepend a column of ones to X
    w = torch.zeros(d + 1, 1)  # Initialize w

    for _ in range(num_iter):
        y_pred = torch.matmul(X, w)
        error = y_pred - Y
        gradient = torch.matmul(X.t(), error) / n
        w = w - lrate * gradient

    return w
    # pass

def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    n, d = X.shape
    X = torch.cat([torch.ones(n, 1), X], dim=1)  # Prepend a column of ones to X
    w = torch.matmul(torch.pinverse(X), Y)
    return w
    # pass

X, Y = utils.load_reg_data()
b, a = linear_normal(X, Y)
line = a * X + b

fig, ax = plt.subplots()
ax.scatter(X.numpy(), Y.numpy(), label='Dataset')
ax.plot(X.numpy(), line.numpy(), 'g-', label='Linear Normal')
ax.set_title('Linear Regression')
ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.legend()
plt.show()
 # pass

# Problem 3
def poly_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float): the learning rate
        num_iter (int): number of iterations of gradient descent to perform

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    n, d = X.shape
    X_new = torch.ones(n, 1 + d + d * (d + 1) // 2)
    X_new[:, 1:d+1] = X
    k = d + 1
    for i in range(d):
        for j in range(i, d):
            X_new[:, k] = X[:, i] * X[:, j]
            k += 1
    
    w = torch.zeros(1 + d + d * (d + 1) // 2, 1)
    for _ in range(num_iter):
        pred = torch.matmul(X_new, w)
        gradient = torch.matmul(X_new.t(), pred - Y) / n
        w -= lrate * gradient
    return w
    pass

def poly_normal(X,Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    n, d = X.shape
    X_new = torch.ones(n, 1 + d + d * (d + 1) // 2)
    X_new[:, 1:d+1] = X
    k = d + 1
    for i in range(d):
        for j in range(i, d):
            X_new[:, k] = X[:, i] * X[:, j]
            k += 1
            
    return torch.matmul(torch.pinverse(X_new), Y)
    # pass

X, Y = utils.load_reg_data()
c, b, a = poly_normal(X, Y)
line = a * X**2 + b * X + c
fig, ax = plt.subplots()
ax.scatter(X.numpy(), Y.numpy(), label='Dataset')
ax.plot(X.numpy(), line.numpy(), 'g-', label='Polynomial Normal')
ax.set_title('Polynomial Regression')
ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.legend()
plt.show()
# pass

def poly_xor():
    '''
    Returns:
        n x 1 FloatTensor: the linear model's predictions on the XOR dataset
        n x 1 FloatTensor: the polynomial model's predictions on the XOR dataset
    '''
    def poly_pred(X, w):
        n, d = X.shape
        X_new = torch.ones(n, 1 + d + d * (d + 1) // 2)
        X_new[:, 1:d+1] = X
        k = d + 1
        for i in range(d):
            for j in range(i, d):
                X_new[:, k] = X[:, i] * X[:, j]
                k += 1
        return torch.matmul(X_new, w)

    X, Y = utils.load_xor_data()
    w_linear = linear_normal(X, Y)
    pred_linear = lambda x: torch.matmul(torch.cat([torch.ones(x.shape[0], 1), x], dim=1), w_linear)
    w_poly = poly_normal(X, Y)
    pred_poly = lambda x: poly_pred(x, w_poly)
    plt.figure(figsize=(7, 4))
    utils.contour_plot(-1, 1, -1, 1, pred_linear) 
    utils.contour_plot(-1, 1, -1, 1, pred_poly)
    return pred_linear(X), pred_poly(X)

poly_xor_preds = poly_xor()
    # pass

