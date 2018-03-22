#pragma once
#include <stdio.h>
#include "Node.h"

enum LayerType
{
	CONV,	//ExtraData length should be 4, depth, convWidth, convStride, imgWidth
	FCL,	//ExtraData length should be 1, with no data elements.
	POOL
};

class Layer
{
public:
	Layer(int nodeCount, int numInputs,LayerType type, ActivationFunction activationFunction, int id, int* extraData);

	//output values and numoutputs must be valid memroy
	void Evaluate(float* inputs, int numInputs, float* outputValues, int* numOutputs);
	int GetNumOutputs();
	LayerType GetType();
	void LogStructure(bool toFile, bool toConsole, FILE* file);
	void LogState(int runId, bool toFile, bool toConsole, FILE* file);
private:
	int id;
	int numNodes;
	Node* nodes;
	LayerType layerType;
};