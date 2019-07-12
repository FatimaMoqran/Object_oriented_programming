import numpy as np
from  create_dataset import Create_Xor
from activation import Activation
from forward import Forward
from backward import Backward

myDataset = Create_Xor()

#myDataset.print_shape()
#myDataset.print_XY()

X =  myDataset.get_X()
Y = myDataset.get_Y()

#creation d'un objet 'annForward'en definissant mes layers
annForward = Forward([2,3,1])

#start loop iterations

learning_rate = 0.0075
num_iterations = 3000
np.random.seed(1)
#track the cost
costs = []
print_cost = True
#boucle de 0 à nombre d'iterations:
    
for i in range (0,num_iterations):
        
    #forward propagation l layers
    annForward.forward_layers(X)
      
    cost=annForward.compute_cost(Y)
    caches = annForward.caches
    parameters = annForward.parameters
    AL = annForward.AL
    
    if i == 0:
        annBackward = Backward(AL, Y, caches, parameters)
        
    annBackward.l_model_backward()
    newParameters = annBackward.update_parameters()
        
    # Print the cost every 100 training example
    if print_cost and i % 1000 == 0:
        print(cost)
    #print (f"Cost after iteration {i}{cost}")
    if print_cost and i % 1000 == 0:
        costs.append(cost)
                