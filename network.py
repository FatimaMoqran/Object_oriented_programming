
import numpy as np
import matplotlib.pyplot as plt
from activation import Activation

class Network(Activation): 
    """
        class that make forward and Backward
    """
    #propriétés

    
    
    def __init__(self,X,Y,dim_layers,learning_rate = 0.00075, num_iterations = 4, print_cost = True):

        """ 
          Function that initialize weights and biais for each layer
          :param dim_layers: list of each layer
          :type dim_layers: pyhon list
          :return parameters dictionnary with W1,b1,......WL,bL
          :rtype: python dictionnary
          Wl --- weight matrix of shape (dim_layers[l],dim_layers[1-l])
          b1 --- weight matrix of shape (dim_layer[l],dim_layers[1-l])
        """
        self.X = X
        self.Y = Y
        self.caches = []
        self.AL  = None
        self.dim_layers = dim_layers
        
        
        np.random.seed(3)
        self.parameters={}
        self.grads = {}
        self.learning_rate = 0.0075
        self.num_iterations = 2
        #keep track of cost
        self.costs = []
        self.print_cost = True
        

        self.layer_nb = len(self.dim_layers)

        for l in range(1,self.layer_nb):
            self.parameters[f'W{l}']= np.random.randn(self.dim_layers[l], self.dim_layers[l-1])*0.01

            self.parameters[f'b{l}'] = np.zeros((self.dim_layers[l],1))

            print(f'l = {l} - W{l}.shape = {self.dim_layers[l]},{self.dim_layers[l-1]}')
#             print(f"b{l}.shape = {self.dim_layers[l]}, 1 ")

            assert (self.parameters[f"W{l}"].shape == (self.dim_layers[l],self.dim_layers[l-1]))
            assert (self.parameters[f"b{l}"].shape == (self.dim_layers[l],1))

        return None

    
    
    def linear_activation_forward (self,A_prev ,W , b, activation):
        """
          function that both computes preactivation and activation 
          :param A_prev: previews activation matrix (for the first layer it is X )
          :param W: weight matrix for the current layer
          :param b: biais matrix for the current layer
          :param activation: choice of activation function eg: sigmoid , relu
          :type A_prev : matrix of float
          :type W: matrix of float
          :type b: matrix of float
          :return: A matrix of activation
          :cache: tuple of (linear_cache,activation_cache) 
          """

        Z = np.dot(W,A_prev)+b

        linear_cache = (A_prev,W,b)


        assert(Z.shape == (W.shape[0], A_prev.shape[1]))

        if activation == "sigmoid":

            A = self.sigmoid(Z)
            activation_cache = A

        elif activation == "relu":

            A = self.relu(Z)
            activation_cache = A


        assert (A.shape == Z.shape)

        cache = (linear_cache, activation_cache)

        return A,cache

    def forward_layers(self):
        """
           Function that computes the forward activation for L layers
          :param X: input matrix, shape (input_size, number of exemple)
          :param parameters: output of initialisation_deep dictionnary of W, b
          :type X: matrix of float
          :type parameters : dictionary of matrices
          :return AL : last post activation value
          :return caches : list of caches with every caches of linear_activation_forward 
          :rtype AL: matrix of float
          :rtype caches: list of tuples

      """
        #je crée une  liste de  caches où je vais stoker les valeurs obtenues
        L = self.layer_nb -1
        print(L)
        A = self.X
        # je calcul le nombre de couches par rapport aux nombres de paramètres
        #je fais boucle pour toutes les couches jusqu'à L-1
        for l in range (1,L): 
            print("l =", l)
        #je considère X comme 
            A_prev= A
    #       je fais mon calcul pour A0 jusqu'à L-1 ou bien (1 à L)
            A,cache = self.linear_activation_forward(A_prev,self.parameters[f'W{l}'],self.parameters[f'b{l}'],"sigmoid")

            #je rajoute le cache obtenu dans la liste cache
            self.caches.append(cache)

          #calcul pour la dernière couche:
          #je récupère le dernier A qui est sorti de mes couches précédentes et je lui mets une sigmoid 
        
        self.AL,cache = self.linear_activation_forward(A,self.parameters[f'W{L}'],self.parameters[f'b{L}'],"sigmoid")
        self.caches.append(cache)

        assert self.AL.shape == (1,self.X.shape[1])

        return None

       
    def compute_cost(self):
        """
            Function that compute the cost 
            :param AL: probability vector - shape (1,number of examples)
            :param Y:  matrix of float
            :type AL: matrix
            :type Y: array of booleen
            :return cost: cost result
            :rtype: float 
        """
        m = self.Y.shape[1]

        #je calcule d'abord llog
        logprob = (self.Y * np.log(self.AL) + (1-self.Y) * np.log(1-self.AL))
        #ensuite la cost
        cost = -(np.sum(logprob))/m
        #je veux que l'on me retourne un nombre et non pas un array
        cost = np.squeeze(cost)
        #être sur que j'ai la cost au bon format
        assert(isinstance(cost,float))

        return cost


    def linear_backward(self,dZ,cache):
        """
            function that computes the linear backward
            :param dZ: gradient of the cost with respect to linear output
            :param cache: tuple of value
            :return dA_prev: gradient of the cost with respect to activation
            :return dW: gradient of the cost with respect to W
            :return db: gradient of the cost with respect to b
        """
        #recuperation des valeurs dont j'ai besoin

        A_prev, W, b = cache
        m=A_prev.shape[1]
        #calcul des dérivées:

        dW = np.dot(dZ,A_prev.T)/m
        db = np.sum(dZ,axis=1,keepdims = True)/m
        print(W.shape, dZ.shape)
        dA_prev= np.dot(W.T,dZ)

        assert(dA_prev.shape == A_prev.shape)
        assert(dW.shape == W.shape)
        assert(db.shape == b.shape)

        return dA_prev,dW,db    
    
    def linear_activation_backward(self,dA,cache,activation):
        """  
            function that computes the linear activation backward with sigmoid and relu
            :param dA: gradient of the activation for the current layer the l layer
            :param cache: tuple of values with the parameters
            :param activation: activation function-Relu or Sigmoid
            :type dA: numpy array
            :type activation: string
            :return dA_prev: gradient activation of the l-1 layer-shape = A_prev shape
            :return dW: gradient of the cost with respect of the W for the current layer l
            :return db: gradient of the cost with respect of the b for the current layer l
        """
        #je récupère de mon cache les paramètres 

        linear_cache, activation_cache = cache

        # je calcule dZ en fonction de l'activation RELU ou Sigmoid
        if activation == 'relu':
            dZ = self.derivative_relu(dA, activation_cache)
            dA_prev,dW,db = self.linear_backward (dZ,linear_cache)

        elif activation == 'sigmoid':
            dZ = self.derivative_sigmoid(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ,linear_cache)

        return dA_prev, dW, db
    
    
    def l_model_backward(self):

        """ 
                function that compute the backward propagation
                :param AL: array with the last activation -probability vector 
                :param Y: array with the label
                :param caches: list of caches of all the parameters of relu activation and one cache with all    the parameters with sigmoid
                :type AL: numpy array
                :type Y: vectord
                :type caches: python list
                :return grads: dictionnary of gradients dW,db,dA

#         """

#         print("initialisation de grad")
        L = self.layer_nb-1 #nombre de couches (correspond aux nombres de caches)
        m = self.AL.shape[1]
        self.Y = self.Y.reshape(self.AL.shape) #on reshape Y comme AL pour pouvoir faire les opérations
#         print("L = ", L, " m = ",m, " Y = ",self.Y)
#             initialisation de la back propagation pour calculer dAL


        dAL = -(np.divide(self.Y,self.AL)-np.divide(1-self.Y,1-self.AL))
        #     print("dAL=", dAL)


              #calcul des gradients pour la dernière couche L sigmoid 
            #j'utilise le cache de la dernière couche et je mets tout dans un dictionnaire

        current_cache = self.caches[L-1]
#         print("current_cache = ", current_cache)
        self.grads[f"dA{L-1}"], self.grads[f"dW{L}"], self.grads[f"db{L}"] = self.linear_activation_backward(dAL,current_cache, activation ="sigmoid")
#         print([f"dA{L-1}"], grads[f"dW{L}"], grads[f"db{L}"])

        #ensuite je fais une boucle pour les autres couches de l= l-2 à l = 0

        for l in reversed(range(L-1)):
            """ 
                   entrée : la dérivée dA l+1 et le cache de la couche current
                   sortie : la dérivée dA l et dWl+1 et dbl+1

           """

            current_cache = self.caches[l]

            print("l = ", l , "current_cache W =", current_cache[0][1].shape)
    #         #je crée des variables temporaires: dA_prev_temp, dW_temp, db_temp

            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward (self.grads[f"dA{l+1}"], current_cache, activation = "sigmoid")

        # #         #je les mets dans le dictionnaire

            self.grads[f"dA{l}"] = dA_prev_temp
            self.grads[f"dW{l+1}"] = dW_temp
            self.grads[f"db{l+1}"] = db_temp

#             print ("dA", l, self.grads[f"dA{l}"].shape)
#             print ("dW", l+1, self.grads[f"dW{l+1}"].shape)
#             print ("db", l+1, self.grads[f"db{l+1}"].shape)


        return self.grads
        

    def update_parameters(self):
            """
                function that update parameters using the gradient descent
                :argument parameters: python dictionary with parameters
                :argument grads: python dictionnary with all the gradient
                :return parameters: python dictionnar with the updated parameters
            """
            L = self.layer_nb -1
            for l in range(L):
                self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - self.learning_rate * self.grads["dW" + str(l+1)]

                self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - self.learning_rate * self.grads["db" + str(l + 1)]

            return self.parameters


   
    #fonction globale qui compute tout (on commence pour faire la forward)

    def L_layer_model(self):

        """
            :param X: matrix of inputs
            :param Y: vector of label
            :param layers_dims: list that contains the input size and each layer size
            :param learningrate: learning rate for the gradient descent
            :param num_iterations: number of iterations of the loop
            :param print_cost: decide if it print the cost (True)or not (False) 
            :type X: numpy matrix
            :type Y: numpy array
            :type layers_dims: python list
            :type learningrate float
            :type num_iteration: int
            :type printcost : bool
            :return 
        """
        np.random.seed(1)
        self.costs = []

        #initialisation des paramètres

#         parameters = initialisation_deep(dim_layers)

        #boucle de 0 à nombre d'iterations:

        for i in range (0,self.num_iterations):

            #forward propagation l layers
            self.forward_layers()

            #compute cost
            cost = self.compute_cost()
    #         print(cost)

            #L model backward
            self.grads = self.l_model_backward()

            #update parameters
            self.parameters = self.update_parameters()

        # Print the cost every 100 training example
            if self.print_cost and i % 1000 == 0:
                print(cost)
    #             print (f"Cost after iteration {i}{cost}")
            if self.print_cost and i % 1000 == 0:
                self.costs.append(cost)

    #     plot the cost
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title(f"Learning rate ={self.learning_rate}")
        plt.show()

        return self.parameters
