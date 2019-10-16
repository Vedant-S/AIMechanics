import DeepLearningLibrary.isthara_versatile as ist
import matplotlib.pyplot as plt

def model(X, Y, layers_dims,activation_tuple, learning_rate = 0.01, num_iterations = 15000, print_cost = True):
    grads = {}
    costs = [] # to keep track of the loss
    m = X.shape[1] # number of examples
    
    # Initialize parameters dictionary.
    parameters = ist.intialize_parameters(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        AL, caches,activation_tuple = ist.linear_deep_forward(X, parameters,activation_tuple)
        
        # Loss
        cost = ist.compute_cost(AL, Y)

        # Backward propagation.
        grads = ist.linear_deep_backward(AL, Y, caches,activation_tuple)
        
        # Update parameters.
        parameters = ist.update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
            
    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per thousand)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    
    return parameters
