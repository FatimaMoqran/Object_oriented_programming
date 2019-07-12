#!/usr/bin/env python
# coding: utf-8


import numpy as np 

class Forward(): 
    """
        class that make forward
    """

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
        self.dim_layers = dim_layers
        np.random.seed(3)
        self.parameters={}

        self.L = len(self.dim_layers)

        for l in range(1,self.L):

            self.parameters[f'W{l}']= np.random.randn(self.dim_layers[l], self.dim_layers[l-1])*0.01

            self.parameters[f'b{l}'] = np.zeros((self.dim_layers[l],1))

    #     print(f'W{l}.shape = {dim_layers[l]},{dim_layers[l-1]}')
    #     print(f"b{l}.shape = {dim_layers[l]}, 1 ")

        assert (self.parameters[f"W{l}"].shape == (self.dim_layers[l],self.dim_layers[l-1]))
        assert (self.parameters[f"b{l}"].shape == (self.dim_layers[l],1))

        return None


