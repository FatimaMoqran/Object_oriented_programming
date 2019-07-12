from  create_dataset import Create_Xor
from activation import Activation
from forward import Forward
from backward import Backward

myDataset = Create_Xor()

#myDataset.print_shape()
#myDataset.print_XY()

X =  myDataset.get_X()
annForward = Forward([2,3,1])

#print(annForward.parameters)
parameters = annForward.parameters
print(parameters)
annForward.forward_layers(X)


# print(annForward.AL)
# print(annForward.caches)
Y = myDataset.get_Y()

#get the forward and caches 

cost=annForward.compute_cost(Y)
caches = annForward.caches
parameters = annForward.parameters
AL = annForward.AL
# print(cost)
# print(caches)

annBackward = Backward(AL, Y, caches, parameters)

# print(annBackward.AL)
annBackward.l_model_backward()
newParameters = annBackward.update_parameters()
print(newParameters)