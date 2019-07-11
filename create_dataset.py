import numpy as np

"""
    class Create_Xor()
"""

class Create_Xor():
    """
        Class that creates Xor dataset
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
        
    def print_shape(self):
        
        """
            Function that prints shapes of X and Y
            :var X: nb_features with shape (2,400)
            :var Y: Y true-false labels with shape (400,1) 
        """
        print(f"\nX.shape = {self.X.shape}")
        print(f"Y.shape = {self.Y.shape}\n")

    def print_XY(self):
        
        print ("\nX = ")
        print(self.X)
        #print Y
        
        print("\nY= ")
        print(self.Y)
        print("\n")
        #value returned
        return None
    
    def get_X(self):
        """
            Function that prints values of X and Y
            :var X: nb_features with shape (2,400)
            :var Y: Y true-false labels with shape (400,1)
            
        """
        return self.X
    
    def get_Y(self):
        
        """
            Function that prints values of Y and Y
            :var X: nb_features with shape (2,400)
            :var Y: Y true-false labels with shape (400,1)
        """
        return self.Y
    
    def get_XY(self):
        """
            Function that return values of X and Y
        """
        return self.X, self.Y


 