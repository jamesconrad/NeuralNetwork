#include <math.h>
#include <random>
#include "Layer.h"
#include "Node.h"



Node::Node(int weightCounts, Layer* parentLayer, ActivationFunction activationFunction, int idnum, int* extraData)
{
	id = idnum;
	parent = parentLayer;
	layerType = parentLayer->GetType();

	numWeights = weightCounts;
	weights = new float[weightCounts];
	for (int i = 0; i < numWeights; i++)
		weights[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;

	links = new int[weightCounts];
	for (int i = 0; i < numWeights; i++)
		links[i] = i;

	//ExtraData length should be 6, kernels, filters, convWidth, convStride, imgWidth
	if (layerType == CONV && extraData != nullptr)
	{
		cWidth = extraData[3];
		cStride = extraData[4];
		inputWidth = extraData[5];
	}

	actFunct = activationFunction;
	bias = (float)rand() / RAND_MAX * 2.f - 1.f;
}


float Node::Evaluate(float* inputs, int numInputs)
{
	if (layerType == FCL)
	{
		float sum = 0;
		for (int i = 0; i < numInputs; i++)
			sum += inputs[links[i]];
		return lastEval = Activation(sum + bias);
	}
	else if (layerType == CONV)
	{
		int totalSteps = 0;
		float* imgResult = new float[totalSteps];

		float sum = 0;
		for (int i = 0; i < numInputs; i++)
			sum += inputs[links[i]];
		return lastEval = Activation(sum + bias);
	}
}


float Node::Activation(float x)
{
	switch (actFunct)
	{
	case SIGMOID:
		return 1.f / (1.f + exp(x));
	case TANH:
		return tanh(x);
	case RELU:
		return fmax(0,x);
	default:
		return 0.f;
	}
}

float Node::DerivativeActivation(float x)
{
	switch (actFunct)
	{
	case SIGMOID: {
		float fx = 1 / (1 + exp(x));
		return fx * (1 - fx); }
	case TANH:
		return 1 - pow(tanh(x), 2);
	case RELU:
		return x >= 0.f ? 1.f : 0.f;
		return 0.f;
	default:
		return 0.f;
	}
}

char* ActivationFunctionString(ActivationFunction a)
{
	switch (a)
	{
	case SIGMOID: return "SoftStep";
	case TANH: return "tanh";
	case RELU:	return "ReLu";
	default: return "ERR";
	}
}

void Node::LogStructure(bool toFile, bool toConsole, FILE* file)
{

}

void Node::LogState(int runId, bool toFile, bool toConsole, FILE* file)
{
	//CSV Format is
	//RunID, LayerID, LayerType, NueronID, ActivationFunction, LastValue, [Weights], Bias
	//At this point in time, The current line in the file should be "RunID, LayerID, LayerType,"
	if (toFile)
	{
		fprintf(file, "%i,%s,%g,\"[", id, ActivationFunctionString(actFunct),lastEval);
		for (int i = 0; i < numWeights - 1; i++)
			fprintf(file, "%g,", weights[i]);
		fprintf(file, "%g]\",%g\n", weights[numWeights - 1], bias);
	}
	if (toConsole)
	{
		printf("%i,%s,%g,\"[", id, ActivationFunctionString(actFunct),lastEval);
		for (int i = 0; i < numWeights - 1; i++)
			printf("%g,", weights[i]);
		printf("%g]\",%g\n", weights[numWeights - 1], bias);
	}
}