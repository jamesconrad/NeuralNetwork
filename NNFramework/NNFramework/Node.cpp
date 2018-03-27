#include <math.h>
#include <random>
#include "NeuralNetwork.h"
#include <time.h>
//#include "Layer.h"
//#include "Node.h"

Node::Node(int weightCounts, Layer* parentLayer, ActivationFunction activationFunction, int idnum, int* extraData)
{
	id = idnum;
	parent = parentLayer;
	layerType = parentLayer->GetType();

	numWeights = weightCounts;
	weights = new float[weightCounts];
	//weights.resize(weightCounts);
	srand(time(NULL));
	for (int i = 0; i < numWeights; i++)
		weights[i] = ((float)rand() / RAND_MAX) * 1.f;//2.f - 1.f;
		//weights[i] = 0.5f;

	//links = new int[weightCounts];
	//for (int i = 0; i < numWeights; i++)
	//	links[i] = i;

	actFunct = activationFunction;
	bias = 0;
	//bias = (float)rand() / RAND_MAX * 2.f - 1.f;
}


float Node::Evaluate(float* inputs, int numInputs)
{
	float sum = 0;
	for (int i = 0; i < numInputs; i++)
		sum += inputs[i] * weights[i];
	lastSum = sum + bias;
	float t = lastEval = Activation(sum + bias);
	return t;
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

float Node::Error(float target)
{
	return 0.5f * pow(target - lastEval, 2);
}

float Node::Backpropogate(float target)
{
	if (parent->next == NULL)
	{
		float targetwrtout = -(target - lastEval);
		float outwrtnet = lastEval * (1 - lastEval);
		float nodeDelta = targetwrtout * outwrtnet;
		error = nodeDelta;
		for (int i = 0; i < numWeights; i++)
		{
			//how impactful was weight i - following the delta rule https://en.wikipedia.org/wiki/Delta_rule
			float errorwrtweight = nodeDelta * parent->prev->nodes[i].lastEval;
			weights[i] = weights[i] - LEARNING_RATE * errorwrtweight;
		}
		return error;
	}
	else
	{
		//calculate this nodes error
		error = 0;
		for (int i = 0; i < parent->next->numNodes; i++)//error = summation(nodeDelta * connectionWeight)
			error += parent->next->nodes[i].error * parent->next->nodes[i].weights[id];

		float outwrtnet = lastEval * (1 - lastEval);
		float nodeDelta = error * outwrtnet;
		error = nodeDelta;

		for (int i = 0; i < numWeights; i++)
		{
			float errorweight = nodeDelta * (parent->prev != NULL ? parent->prev->nodes[i].lastEval : parent->nn->lastInput[i]);
			weights[i] = weights[i] - LEARNING_RATE * errorweight;
		}

		return error;
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