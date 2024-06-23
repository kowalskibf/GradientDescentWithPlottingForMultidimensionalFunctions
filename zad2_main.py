import cec2017
import numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from cec2017.functions import f1, f2, f3

MAX_X = 100
PLOT_STEP = 0.1

UPPER_BOUND = 100
DIMENSIONALITY = 2

def booth(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

def gradient_descent(function, initial_point, learning_rate, max_iterations):
    function_gradient = grad(function)
    current_point = initial_point
    trajectory = [current_point]
    for _ in range(max_iterations):
        gradient = function_gradient(current_point)
        current_point -= learning_rate * gradient
        current_point = np.clip(current_point, -100, 100)
        trajectory.append(current_point)
        print(f'Function {fi} Iteration {_} ',[xi for xi in current_point])
    return [current_point, trajectory]

functions = [booth, f1, f2, f3]
dimensions = [2, 10, 10, 10]
learning_rates = [0.1, 10e-9, 10e-20, 10e-10]

results = []

for fi in range(len(functions)):
    UPPER_BOUND = 100
    DIMENSIONALITY = dimensions[fi]
    ok = False
    while not ok:
        try:
            x = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=DIMENSIONALITY)
            initial_point = x
            max_iterations = 1000
            beta = 1
            [final_position, trajectory] = gradient_descent(functions[fi], initial_point, learning_rates[fi], max_iterations)
            ok = True
        except:
            print("Again...")
        
    x_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    y_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.empty(X.shape)
    q=functions[fi]
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = q(np.array([X[i, j], Y[i, j]]))
    plt.contour(X, Y, Z, 20)
    for i in range(1, len(trajectory)):
        plt.arrow(trajectory[i-1][0], trajectory[i-1][1], trajectory[i][0] - trajectory[i-1][0], trajectory[i][1] - trajectory[i-1][1], head_width=0.1, color='black')
    if fi == 0:
        title = 'Booth'
    else:
        title = 'CEC 2017 f' + str(fi)
    title += ', f(x_final) = ' + str(functions[fi](final_position))
    plt.title(title)

    plt.savefig(f'img/{fi}.png')
    plt.clf()
    results.append(final_position)

for result in results:
    print(result)