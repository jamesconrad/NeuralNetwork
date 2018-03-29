#pragma once
#include <stdio.h>
#include "Node.h"

enum LayerType
{
	CONV,
	FCL,
	POOL
};

class NeuralNetwork;

class Layer
{
public:
	friend Node;
	Layer(int nodeCount, int numInputs,LayerType type, ActivationFunction activationFunction, int id, int* extraData);

	//output values and numoutputs must be valid memroy
	void Evaluate(float* inputs, int numInputs, float* outputValues, int* numOutputs);
	void Backpropogate(float* target = nullptr);

	int GetNumOutputs();
	LayerType GetType();
	void LogStructure(bool toFile, bool toConsole, FILE* file);
	void LogState(int runId, bool toFile, bool toConsole, FILE* file);
	Layer* prev;
	Layer* next;
	NeuralNetwork* nn;

	void ReLink()
	{
		for (int i = 0; i < numNodes; i++)
			nodes[i].parent = this;
	}

	void SetNode(int id, std::vector<float> &weights, float bias)
	{
		nodes[id].bias = bias;
		for (int i = 0, s = weights.size(); i < s; i++)
		{
			nodes[id].weights[i] = weights[i];
		}
	}

private:
	int id;
	int numNodes;
	Node* nodes;
	//std::vector<Node> nodes;
	LayerType layerType;
};