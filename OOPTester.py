from  create_dataset import Create_Xor
from activation import Activation
from forward import Forward

myDataset = Create_Xor()

#myDataset.print_shape()
#myDataset.print_XY()

X =  myDataset.get_X()
annForward = Forward([2,3,1])

#print(annForward.parameters)
parameters = annForward.parameters
# print(parameters)
annForward.forward_layers(X)


# print(annForward.AL)
# print(annForward.caches)
Y = myDataset.get_Y()

#get the forward and caches 
cost=annForward.compute_cost(Y)

caches = annForward.caches
print(cost)
print(caches)
