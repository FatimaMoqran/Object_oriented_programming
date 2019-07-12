import numpy as np
from activation import Activation
from forward import Forward 

class Backward(Forward):
    
    """
        class that computes backward propagation
    """
    def __init__(self, AL, Y, caches, parameters):
        
        self.AL = AL
        self.Y = Y
        self.caches = caches
        self.parameters = parameters    
         
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
        dA_prev= np.dot(W.T,dZ)

        assert(dA_prev.shape == A_prev.shape)
        assert(dW.shape == W.shape)
        assert(db.shape == b.shape)
        return dA_prev,dW,db
    
    # fonction backward pour l'implémentation de la backward si fonction d'activation est une sigmoid ou fonction d'activation est une RELU

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
    
    def l_model_backward(self,AL,Y,caches):

        """ 
              function that compute the backward propagation
              :param AL: array with the last activation -probability vector 
              :param Y: array with the label
              :param caches: list of caches of all the parameters of relu activation and one cache with all the parameters with sigmoid
              :type AL: numpy array
              :type Y: vectord
              :type caches: python list
              :return grads: dictionnary of gradients dW,db,dA

        """
        grads = {}
    #     print("initialisation de grad")
        L = len(self.caches) #nombre de couches (correspond aux nombres de caches)
        m = self.AL.shape[1]
        self.Y = self.Y.reshape(self.AL.shape) #on reshape Y comme AL pour pouvoir faire les opérations
    #     print("L = ", L, " m = ",m, " Y = ",Y)
        #     initialisation de la back propagation pour calculer dAL


        dAL = -(np.divide(Y,self.AL)-np.divide(1-Y,1-self.AL))
    #     print("dAL=", dAL)


          #calcul des gradients pour la dernière couche L sigmoid 
        #j'utilise le cache de la dernière couche et je mets tout dans un dictionnaire

        current_cache = self.caches[L-1]
    #     print("current_cache = ", current_cache)
        grads[f"dA{L-1}"], grads[f"dW{L}"], grads[f"db{L}"] = self.linear_activation_backward(dAL,current_cache, activation ="sigmoid")
    #     print([f"dA{L-1}"], grads[f"dW{L}"], grads[f"db{L}"])

        #ensuite je fais une boucle pour les autres couches de l= l-2 à l = 0

        for l in reversed(range(L-1)):
            """ 
    #             entrée : la dérivée dA l+1 et le cache de la couche current
    #             sortie : la dérivée dA l et dWl+1 et dbl+1

    #         """

            current_cache = self.caches[l]

    #         print("l = ", l , "current_cache W =", current_cache[0][1].shape)
    #         #je crée des variables temporaires: dA_prev_temp, dW_temp, db_temp

            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward (grads[f"dA{l+1}"], current_cache, activation = "sigmoid")

    # #         #je les mets dans le dictionnaire

            grads[f"dA{l}"] = dA_prev_temp
            grads[f"dW{l+1}"] = dW_temp
            grads[f"db{l+1}"] = db_temp

    #         print ("da", l, grads[f"dA{l}"].shape)


        return grads

    def update_parameters(grads,learning_rate):
        """
            function that update parameters using the gradient descent
            :argument parameters: python dictionary with parameters
            :argument grads: python dictionnary with all the gradient
            :return parameters: python dictionnar with the updated parameters
        """
        L = len (self.parameters)//2
        for l in range(L):
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]

            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l + 1)]

        return parameters

    