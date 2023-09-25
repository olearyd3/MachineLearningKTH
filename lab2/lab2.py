import random
import math
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def Precompute(X, Y, N, Kernel, t):
    global PreComputedMatrix
    PreComputedMatrix = np.ndarray(shape=(N, N))
    PreComputedMatrix = [
        [t[i] * t[j] * Kernel(X[i], Y[j]) for j in range(N)] for i in range(N)]

    # return PreComputedMatrix


# finds a vector (alpha) to minimize the function 'objective' within bounds B and constraints XC
# """ ret = minimize(objective, start, bounds=B, constraints=XC)
# alpha = ret['x'] """

def LinearKernel(X, Y):
    return np.dot(X, Y)


def PolyNomialKernel(X, Y, power=3):
    return np.power(np.dot(X, Y) + 1, power)

 # np.power(np.dot(x, y) + 1, p)


def RBFKernel(X, Y):
    # return math.exp(-np.linalg.norm(X-Y, 2)**2/(2*sigma**2))
    sigma = 0.5
    diff = X - Y
    return np.exp(-np.dot(diff, diff) / (2.0 * np.power(sigma, 2)))

# Implements equation 4 - 1/2SUMSUM alphaalpha tt Kernereturn numpy.dot(numpy.transpose(X), Y)l - SUMalpha. Receives alpha as a parameter


def objective(alpha):
    #print(0.5 * np.sum(np.outer(alpha, alpha) * PreComputedMatrix) - np.sum(alpha))
    return 0.5 * np.sum(np.outer(alpha, alpha) * PreComputedMatrix) - np.sum(alpha)

# implements the equality constraint of eqn 10: 0 <= alpha_i <= C and SUM alpha_i t_i = 0


def zerofun(alpha):
    return np.dot(alpha, targets)

# nonzero = [(alpha[i], inputs[i], targets[i]) for i in range(N) if abs(alpha[i]) > 10e-5]
# use equation 7 and a point on the margin that corresponds with a point with alpha larger than 0 but less than C


def calculateB(alpha, N):
    global nonzero
    nonzero = [(alpha[i], inputs[i], targets[i])
               for i in range(N) if abs(alpha[i]) > 10e-5]
    bsum = 0
    for value in nonzero:
        bsum += value[0] * value[2] * Kernel(value[1], nonzero[0][1])
    return bsum - nonzero[0][2]
# eqn 6 -- uses the   non-zero alphas together with their x_is and t_is to classify new points


# def indicator(s, alpha, t, X, b, kernel):
#     K = np.array([kernel(s, X[i]) for i in range(X.shape[0])])
#     return np.dot(alpha * t, K) - b

def indicator(x, y, b):
    totsum = 0
    for value in nonzero:
        totsum += value[0] * value[2] * Kernel([x, y], value[1])
    return totsum - b


def generateData():
    np.random.seed(100)
    numPoints = 10
    stdDev = 0.2
    upperBoundA = 1.5
    lowerBoundA = 0.5
    upperBoundB = 1.5
    lowerBoundB = 0.5
    global classA
    global classB
    classA = np.concatenate((np.random.randn(10, 2) * 0.3 + [1.5, 0.5],
                             np.random.randn(10, 2)*0.3 + [-1.5, 0.5]))
    classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate(
        (np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

    N = inputs.shape[0]  # number of rows

    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    return inputs, targets, N


def plot(b):
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.', label="Class A")
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.', label="Class B")

    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)
    grid = np.array([[indicator(x, y, b) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
                colors=('red', 'black', 'blue'), linewidths=(1, 2, 1))
    plt.title("Decision Boundary for an SVM with a Linear Kernel")

    custom_lines = [Line2D([0], [0], color='b', lw=1),
                Line2D([0], [0], color='r', lw=1),
                Line2D([0], [0], color='k', lw=2)]

    plt.legend(custom_lines, ['Class A ', 'Class B', 'Decision Boundary'], loc="upper left")
    plt.xlabel("X values")
    plt.ylabel("Y values")
#plt.legend(loc="upper left")
    plt.axis('equal')  # force same scale on both axes
    # plt.savefig('svmplot.pdf')  # save a copy as a pdf
    plt.show()  # show the plot on the screen


def main():

    start = np.zeros(N)  # N is the number of training samples
    C = 10000

    # To have an upper constraint
    B = [(0, C) for b in range(N)]

    # To only have a lower bound:
    # B = [(0, None) for b in range(N)]
    # XC is the second half of equation 10; given as a dictionary with fields type and fun
    # In the following example, zerofun is a function which calculates the value which
    # ...should be constrained to zero.
    XC = {'type': 'eq', 'fun': zerofun}

    ret = minimize(objective, start, bounds=B, constraints=XC)

    if (not ret['success']):  # The string 'success' instead holds a boolean representing if the optimizer has found a solution
        raise ValueError('Cannot find optimizing solution')

    alpha = ret['x']
    b = calculateB(alpha, N)
    plot(b)

    return b

    # print("b value: ", b)
    # plot(b)


if __name__ == "__main__":
    # Define the global variables
    global Kernel
    kernels = [LinearKernel, PolyNomialKernel, RBFKernel]
    # # Kernel = LinearKernel
    # Kernel = PolyNomialKernel
    Kernel = kernels[0]

    inputs, targets, N = generateData()
    Precompute(inputs, inputs, N, Kernel, targets)

    # call main
    main()

    # data = []
    # for kernel in kernels:
    #     Kernel = kernel
    #     Precompute(inputs, inputs, N, Kernel, targets)
    #     data.append(main()):w

    # xgrid = np.linspace(-5, 5)
    # ygrid = np.linspace(-4, 4)
    # # grid = np.array([[indicator(x, y, b) for x in xgrid] for y in ygrid])
    # # plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
    # #             colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

    # fig, axs = plt.subplots(2, 2)
    # Kernel = kernels[0]
    # axs[0, 0].plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    # axs[0, 0].plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    # grid = np.array([[indicator(x, y, data[0]) for x in xgrid] for y in ygrid])
    # axs[0, 0].contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
    #                   colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

    # Kernel = kernels[1]
    # axs[0, 1].plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    # axs[0, 1].plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    # grid = np.array([[indicator(x, y, data[1]) for x in xgrid] for y in ygrid])
    # axs[0, 1].contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
    #                   colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

    # Kernel = kernels[2]
    # axs[1, 0].plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    # axs[1, 0].plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    # grid = np.array([[indicator(x, y, data[2]) for x in xgrid] for y in ygrid])
    # axs[1, 0].contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
    #                   colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

    # axs[1, 1].plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    # axs[1, 1].plot([p[0] for p in classB], [p[1] for p in classB], 'r.')

    # for ax in axs.flat:
    #     ax.set(xlabel='x-label', ylabel='y-label')

    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

    # plt.show()
    # plt.savefig('svmplot.pdf')  # save a copy as a pdf
