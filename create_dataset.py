import numpy as np

"""
    class Create_Xor()
"""

class Create_Xor():
    
    """
        Function that creates XOR dataset
        :param nb_features: number of features
        :type nb_features : int
        :return: X  input numpy array shape(2,400)
        :rtype: numpy array
        :return: Y true-false labels numpy array shape(400,1)
        :rtype: numpy array
    """
    
    def __init__(self, nb_features=400):
        
        """
        Function that creates XOR dataset
        :param nb_features: number of features
        :type nb_features : int
        :return: X  input numpy array shape(2,400)
        :rtype: numpy array
        :return: Y true-false labels numpy array shape(400,1)
        :rtype: numpy array
        """
        
        #nb_features parameter
        self.nb_features = nb_features
        
        #make X features
        self.X = np.random.randint(2, size = (2,self.nb_features))
        #make Y labels
        self.Y = np.logical_xor(self.X[0,:],self.X[1,:])
        #reshape Y
        self.Y = self.Y.reshape(1,nb_features)
        

       


 