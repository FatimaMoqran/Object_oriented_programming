import numpy as np

class Activation():
    
    """
        Class that makes activations
    """

    def sigmoid(Z):
        """
            Function that implement sigmoid
            :param Z: input value
            :type Z: float, array
            :return s: sigmoid result of Z 
            :rtype s:  float or array
        """

        s = 1/(1+(np.exp(-Z)))
      # sigmoid(0) == 0.5

        assert(0.5 == 1/(1+(np.exp(0))))

        return s

    def relu(Z):
        """
              Function that implement relu
              :param Z: input value
              :type Z: float, array
              :return: relu result of Z 
              :rtype: float or array
          """

     #   Z>0 c'est un test si Z>0 alors on va avoir 1 sinon on va avoir 0

        r = Z * (Z>0)

        #r = np.maximum(0,Z)

        assert(0.5 == 0.5 * (0.5>0))
        assert(0   == -1  * (-1>0))


        return r

    def derivative_sigmoid(dA,activation_cache):
        """
          Function that implement the derivative of sigmoid
          :param Z: input value
          :type Z: float, array
          :return: derivative result of Z 
          :rtype: float or array
        """
        Z = activation_cache
    #     print("Z.shape = ", Z.shape, "dA.shape = ", dA.shape)
        s = 1/(1+np.exp(-Z))
        dZ= dA * s *(1-s)
    #     print("dZ.shape = ", dZ.shape, "s.shape = ", s.shape)
        assert(0.25 == sigmoid (0) * (1-sigmoid(0)))

        return dZ
    
    def derivative_relu(dA, activation_cache):
    
        Z = activation_cache
        dZ = np.array(dA,copy = True)
        dZ[Z<= 0] = 0
        assert (dZ.shape == Z.shape)

        return dZ

