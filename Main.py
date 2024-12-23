import math
import random
import time

inputNeuronCount = 5
outputNeuronCount = 5
hiddenNeuronCount = 5

learningRate = 0.1
epochs = 1000
expectedOutputs = [0,1,0,0,0.5]

inputLayer = []
hiddenLayer1 = []
hiddenLayer2 = []
outputLayer = []
layers = [inputLayer, hiddenLayer1, hiddenLayer2, outputLayer]

def sigmoid(x):
    return (1 / (1 + math.exp(-x)))

class Neuron:
    def __init__(self, layer, index, bias):
        self.lastWeights = []
        self.bias = bias
        self.value = 0
        self.index = index
        self.layer = layer
        self.dCdZ = 0

    def activateValue(self):
        self.value = 0
        for neuron in layers[self.layer-1]:
            self.value += neuron.value * self.lastWeights[neuron.index]
        self.value += self.bias
        self.value = sigmoid(self.value)

    def getDZDZ(self, neuron2):
        return self.lastWeights[neuron2.index] * neuron2.value * (1 - neuron2.value)

    def getDCDZ(self, expectedValue):
        return (2 * (self.value - expectedValue)) * self.value * (1 - self.value)



# INPUT LAYER
for i in range(0,inputNeuronCount):
    neuron = Neuron(0,i, 0)
    neuron.value = float(input("Input Layer Neuron "+ str(i) + " value:"))
    inputLayer.append(neuron)


# HIDDEN LAYER 1
for i in range(0,hiddenNeuronCount):
    neuron = Neuron(1,i, random.uniform(-1, 1))
    for j in range(0,inputNeuronCount):
        neuron.lastWeights.append(random.uniform(-1,1))

    neuron.activateValue()
    hiddenLayer1.append(neuron)

# HIDDEN LAYER 2
for i in range(0,hiddenNeuronCount):
    neuron = Neuron(2,i, random.uniform(-1, 1))
    for j in range(0,hiddenNeuronCount):
        neuron.lastWeights.append(random.uniform(-1,1))

    neuron.activateValue()
    hiddenLayer2.append(neuron)

# OUTPUT LAYER
for i in range(0,outputNeuronCount):
    neuron = Neuron(3,i, random.uniform(-1, 1))
    for j in range(0,hiddenNeuronCount):
        neuron.lastWeights.append(random.uniform(-1,1))

    neuron.activateValue()
    outputLayer.append(neuron)

# VISUALIZATION


# CALCULATING COST FUNCTION

def getCost(outputLayer, expectedOutputs):
    cost = 0
    for i in range(len(outputLayer)):
        cost += (outputLayer[i].value - expectedOutputs[i]) ** 2
    return cost


def dCdB(neuron):
    value = 0
    nextLayer = neuron.layer+1
    nextNeurons = []
    while nextLayer != len(layers):
        nextNeurons.append(layers[nextLayer])
        nextLayer += 1

    if len(nextNeurons) >= 1:
        for i in range(len(nextNeurons[0])):
            if len(nextNeurons) >= 2:
                for j in range(len(nextNeurons[1])):
                    value += nextNeurons[0][i].getDZDZ(neuron) * nextNeurons[1][j].getDZDZ(neuron) * outputLayer[j].getDCDZ(expectedOutputs[j])
            else:
                value += outputLayer[i].getDCDZ(expectedOutputs[i]) * nextNeurons[0][i].getDZDZ(neuron)
    else:
        value += neuron.getDCDZ(expectedOutputs[neuron.index])
    return value

def dCdW(neuron, previous):
    value = 0
    nextLayer = neuron.layer + 1
    nextNeurons = []
    while nextLayer != len(layers):
        nextNeurons.append(layers[nextLayer])
        nextLayer += 1

    if len(nextNeurons) >= 1:
        for i in range(len(nextNeurons[0])):
            if len(nextNeurons) >= 2:
                for j in range(len(nextNeurons[1])):
                    value += previous.value * nextNeurons[0][i].getDZDZ(neuron) * nextNeurons[1][j].getDZDZ(neuron) * outputLayer[
                        j].getDCDZ(expectedOutputs[j])
            else:
                value += previous.value * outputLayer[i].getDCDZ(expectedOutputs[i]) * nextNeurons[0][i].getDZDZ(neuron)
    else:
        value += previous.value * neuron.getDCDZ(expectedOutputs[neuron.index])
    return value


for epoch in range(epochs):
    for layer in layers[1:]:
        for neuron in layer:
            neuron.activateValue()

    # Print the current output for visualization
    for i in range(0,5):
        print(f"{inputLayer[i].value}\t{hiddenLayer1[i].value}\t{hiddenLayer2[i].value}\t{outputLayer[i].value}")

    # Backpropagate and update biases/weights
    for i in range(hiddenNeuronCount):
        hiddenLayer1[i].bias -= learningRate*dCdB(hiddenLayer1[i])
        hiddenLayer2[i].bias -= learningRate*dCdB(hiddenLayer2[i])

    for i in range(outputNeuronCount):
        outputLayer[i].bias -= learningRate*dCdB(outputLayer[i])

    for layer in layers[1:]:
        for neuron in layer:
            for previous in layers[neuron.layer-1]:
                neuron.lastWeights[previous.index] -= learningRate*dCdW(neuron, previous)


    print("Loss:", getCost(outputLayer, expectedOutputs))
    print("-------------------------------------------------------------")
