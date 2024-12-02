import math
import random

def sigmoid(x):
    return (1 / (1 + math.exp(-x)))

class Neuron:

    def __init__(self, layer, index):
        self.nextWeights = []
        self.bias = 0
        self.value = 0
        self.index = index
        self.layer = layer

    def activateValue(self, lastLayer):
        self.value = 0
        for neuron in lastLayer:
            self.value += neuron.value*neuron.nextWeights[self.index]
        self.value += self.bias
        self.value = sigmoid(self.value)


# INPUT LAYER
inputLayer = []
for i in range(0,5):
    neuron = Neuron("input",i)
    neuron.value = random.random()
    neuron.bias = random.uniform(-1,1)
    for j in range(0,5):
        neuron.nextWeights.append(random.uniform(-1,1))

    inputLayer.append(neuron)


# HIDDEN LAYER 1
hiddenLayer1 = []
for i in range(0,5):
    neuron = Neuron("hidden1",i)
    neuron.activateValue(inputLayer)
    neuron.bias = random.uniform(-1, 1)
    for j in range(0,5):
        neuron.nextWeights.append(random.uniform(-1,1))
    hiddenLayer1.append(neuron)

# HIDDEN LAYER 2
hiddenLayer2 = []
for i in range(0,5):
    neuron = Neuron("hidden2",i)
    neuron.activateValue(hiddenLayer1)
    neuron.bias = random.uniform(-1, 1)
    for j in range(0,5):
        neuron.nextWeights.append(random.uniform(-1,1))
    hiddenLayer2.append(neuron)

# OUTPUT LAYER
outputLayer = []
for i in range(0,5):
    neuron = Neuron("output",i)
    neuron.activateValue(hiddenLayer2)
    neuron.bias = random.uniform(-1, 1)
    outputLayer.append(neuron)



# VISUALIZATION
for i in range(0,5):
    print(inputLayer[i].value,"\t",hiddenLayer1[i].value,"\t",hiddenLayer2[i].value,"\t",outputLayer[i].value,"\n")

print("-------------------------------------------------------------------------------------")


for neuron in inputLayer:
    neuron.value = 1

for neuron in hiddenLayer1:
    neuron.activateValue(inputLayer)

for neuron in hiddenLayer2:
    neuron.activateValue(hiddenLayer1)

for neuron in outputLayer:
    neuron.activateValue(hiddenLayer2)

for i in range(0,5):
    print(inputLayer[i].value,"\t",hiddenLayer1[i].value,"\t",hiddenLayer2[i].value,"\t",outputLayer[i].value,"\n")