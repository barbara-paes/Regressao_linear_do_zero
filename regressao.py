print("Barbara Fernandes Paes")

import numpy as np
import matplotlib.pyplot as plt

def f_true(x):
    return 2 + 0.8 * x


xs = np.linspace(-3, 3, 100)

ys = np.array([f_true(x) + np.random.randn() * 0.5 for x in xs])

theta = 0
cs = 0
i = 5000 # number of iterations to perform gradient descent
L = 0.0001  # Learning rate
m = float(len(xs))

def h(x, theta, c):
    h = x*theta + c
    return h

def J(theta,c, x, y):
    J = np.sum((h(x,theta,c)-y)**2)/(2*m)
    return J


def gradient(i, theta,c, x, y,L):
    first_iteration = int(i / 2)  # Iteração do gráfico do meio
    past_cost = [0] * i

    for iteration in range(i):
        D_theta = (np.sum((h(x,theta,c)-y)*x))/m
        D_c = (np.sum(h(x,theta,c)-y))/m
        theta = theta - (L*D_theta)
        c = c -(L*D_c)
        cost = J(theta,c,x,y)
        past_cost[iteration] = cost
        

        if (iteration == 0) or (iteration == first_iteration) or (iteration == i-1):
            plt.plot(x, h(x, theta, c), color='b')
            plt.scatter(x, y, color='r', label='Dados de treinamento')
            plt.plot(xs, f_true(xs), color='green', label='2 + 0.8 * x')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.show()



    return theta, past_cost,c

theta_final,past_cost,C = gradient(i,theta,cs,xs,ys,L)
print(theta_final,C)
#print(past_cost)

xs = np.linspace(0, past_cost[-1], len(past_cost))

plt.plot(xs, past_cost, color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.show()