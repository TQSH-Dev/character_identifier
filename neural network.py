import matplotlib.pyplot as plt
import numpy as np
import math as m

def sigmoid(x):
    return 1/(1+(m.e)**(-x))


neurallayers=[]
numoflays=4

class neuron:
    def __init__(self,prevlayer,nextlayer,prevn):
        self.prevlayer=prevlayer
        self.nextlayer=nextlayer
        self.prevn=prevn
        self.bias=1
        if prevn!=0:
            self.weights=np.array([1 for i in range(prevn)])
        else:
            self.weights=None

    def calcactivity(self,activities):
        return sigmoid (self.weights*activities+self.bias)

class neurallayer:
    def __init__(self,prevlayer,nextlayer,prevn,currentn):
        self.prevlayer=prevlayer
        self.nextlayer=nextlayer
        self.currentn=currentn
        self.neurons=[neuron(self.prevlayer,self.nextlayer,prevn) for i in range(currentn)]

neurallayers.append(neurallayer(None,1,0,784))
for i in range(1,numoflays-1):
    neurallayers.append(neurallayer(i-1,i,neurallayers[i-1].currentn,16))
neurallayers.append(neurallayer(numoflays-2,None,neurallayers[numoflays-2].currentn,10))

count=0
for i in neurallayers[1:]:
    for j in i.neurons:
        count+=len(j.weights)+1
print(count)






        
        

