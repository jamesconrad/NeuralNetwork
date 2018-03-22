#include "Layer.h"

Layer::Layer(int nodeCount, int numInputs, LayerType type, ActivationFunction activationFunction, int idnum, int* extraData)
{
	id = idnum;
	numNodes = nodeCount;
	layerType = type;
	nodes = (Node*) new char[nodeCount * sizeof(Node)];

	//ExtraData length should be 6, kernels, filters, convWidth, convStride, imgWidth
	if (layerType == CONV && extraData != nullptr)
	{
		int inRes = extraData[5] * extraData[5];
		for (int i = 0; i < nodeCount; i++)
			nodes[i] = Node(inRes, this, activationFunction, i, extraData);
	}
	else //if (layerType == FCL)
	{
		for (int i = 0; i < numNodes; i++)
		{
			nodes[i] = Node(numInputs, this, activationFunction, i, extraData);
		}
	}
}

int Layer::GetNumOutputs()
{
	return numNodes;
}

LayerType Layer::GetType()
{
	return layerType;
}

void Layer::Evaluate(float* inputs, int numInputs, float* outputValues, int* numOutputs)
{
	for (int i = 0; i < numNodes; i++)
	{
		outputValues[i] = nodes[i].Evaluate(inputs, numInputs);
	}
	*numOutputs = numNodes;
}

char* LayerTypeString(LayerType t)
{
	switch(t)
	{
	case CONV: return "CONV";
	case FCL: return "FCL";
	default: return "ERR";
	}
}

void Layer::LogStructure(bool toFile, bool toConsole, FILE* file)
{

}

void Layer::LogState(int runId, bool toFile, bool toConsole, FILE* file)
{
	//CSV Format is
	//RunID, LayerID, LayerType, NueronID, ActivationFunction, LastValue, [Weights], Bias
	//At this point in time, the function must set the line to "RunID,LayerID,LayerType," then call the nodes log for each node

	for (int i = 0; i < numNodes; i++)
	{
		if (toFile) fprintf(file, "%i,%i,%s,", runId, id, LayerTypeString(layerType));
		if (toConsole) printf("%i,%i,%s,", runId, id, LayerTypeString(layerType));
		nodes[i].LogState(runId, toFile, toConsole, file);
	}
}