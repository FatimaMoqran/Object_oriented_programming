#!/usr/bin/env python
# coding: utf-8


import numpy as np 
from activation import Activation

class Forward(Activation): 
    """
        class that make forward
    """
    #propriétés
    
    
    def __init__(self,dim_layers):

        """ 
          Function that initialize weights and biais for each layer
          :param dim_layers: list of each layer
          :type dim_layers: pyhon list
          :return parameters dictionnary with W1,b1,......WL,bL
          :rtype: python dictionnary
          Wl --- weight matrix of shape (dim_layers[l],dim_layers[1-l])
          b1 --- weight matrix of shape (dim_layer[l],dim_layers[1-l])
        """
        self.caches = []
        self.AL  = None
        self.dim_layers = dim_layers
        np.random.seed(3)
        self.parameters={}

        self.L = len(self.dim_layers)

        for l in range(1,self.L):

            self.parameters[f'W{l}']= np.random.randn(self.dim_layers[l], self.dim_layers[l-1])*0.01

            self.parameters[f'b{l}'] = np.zeros((self.dim_layers[l],1))

            print(f'W{l}.shape = {self.dim_layers[l]},{self.dim_layers[l-1]}')
            print(f"b{l}.shape = {self.dim_layers[l]}, 1 ")

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

    def forward_layers(self,X):
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
        
        A = X
        L = len(self.parameters) // 2 # je calcul le nombre de couches par rapport aux nombres de paramètres
        #je fais une boucle pour toutes les couches jusqu'à L-1
        for l in range (1,L): 
        #je considère X comme 
            A_prev= A
    #       je fais mon calcul pour A0 jusqu'à L-1 ou bien (1 à L)
            A,cache = self.linear_activation_forward(A_prev,self.parameters[f'W{l}'],self.parameters[f'b{l}'],"sigmoid")

            #je rajoute le cache obtenu dans la liste cache
            self.caches.append(cache)

          #calcul pour la dernière couche:
          #je récupère le dernier A qui est sorti de mes couches précédentes et je lui mets une sigmoid 

        self.AL,cache = self.linear_activation_forward(A,self.parameters[f'W{l+1}'],self.parameters[f'b{l+1}'],"sigmoid")
        self.caches.append(cache)

        assert self.AL.shape == (1,X.shape[1])

        return None

        m = X.shape[1]

    def compute_cost(AL,Y):
        """
            Function that compute the cost 
            :param AL: probability vector - shape (1,number of examples)
            :param Y:  matrix of float
            :type AL: matrix
            :type Y: array of booleen
            :return cost: cost result
            :rtype: float 
        """
        #je calcule d'abord llog
        logprob = (Y * np.log(AL) + (1-Y) * np.log(1-AL))
        #ensuite la cost
        cost = -(np.sum(logprob))/m
        #je veux que l'on me retourne un nombre et non pas un array
        cost = np.squeeze(cost)
        #être sur que j'ai la cost au bon format
        assert(isinstance(cost,float))

        return cost

