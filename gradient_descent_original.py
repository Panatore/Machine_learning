import numpy as np
def yhat(xs):
    '''
    This functions helps to plot fast the linear hypothesis
    Return X_ref and yhat(theta0,theta1) as a list
    - input:
        -X: X data points
        -N: number of points you wish to draw (default 2)
    '''
    return [xs,lambda theta0,theta1: theta1*xs+theta0]

def gradient_descent(X,
                     y,
                     theta0=0,
                     theta1=1,
                     alpha = 0.0001,
                     n = 100000,
                     epsilon=0.001
                    ):
    '''
    All the implementations of Gradient descent algorithm
    n is the max number of iterations
    '''
    
    def cost_function(X,y):
        ret_function = lambda theta0,theta1: (((X*theta1+theta0)-y)**2).sum()/len(X)
        return ret_function

    def grad_theta0(X,y):
        ret_function = lambda theta0,theta1: ((2*(X*theta1+theta0-y))).sum()/len(X)
        return ret_function

    def grad_theta1(X,y):
        def ret_function(theta_0,theta_1):
            return 2*sum((theta_0 + theta_1 * X - y) * X) / len(X)
        return ret_function
    
    J = cost_function(X,y)
    Dtheta0 = grad_theta0(X,y)
    Dtheta1 = grad_theta1(X,y)
    cost = J(theta0,theta1)
    
    converged=False
    l_thetas = [[theta0,theta1]]
    l_cost = [cost]

    for i in range(0,n):
        theta0_next = theta0 - alpha*(Dtheta0(theta0,theta1))
        theta1_next = theta1 - alpha*(Dtheta1(theta0,theta1)) 
        theta0, theta1 = theta0_next, theta1_next
        l_thetas.append([theta0,theta1])
        
        new_cost = J(theta0,theta1)
        DeltaCost = new_cost-cost
        cost = new_cost
        l_cost.append(cost)
        if np.abs(DeltaCost)<epsilon:
            converged=True
            break
    return {
        'cost':l_cost,
        'thetas':l_thetas
        }
