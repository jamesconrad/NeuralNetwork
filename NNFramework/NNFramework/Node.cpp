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
	for (int i = 0; i < numWeights; i++)
	{
		weights[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;
		//weights[i] = 0.000001f + ((float)rand() / RAND_MAX) / 1000.f;
		//printf("%g\n", weights[i]);
	}
	actFunct = activationFunction;
	bias = ((float)rand() / RAND_MAX) * 2.f - 1.f;
}


float Node::Evaluate(float* inputs, int numInputs)
{
	//calculate sum of weights*inputs
	float sum = 0;
	for (int i = 0; i < numInputs; i++)
		sum += inputs[i] * weights[i];
	lastSum = sum + bias;
	return lastEval = Activation(lastSum); //run the sum through the activation
}


float Node::Activation(float x)
{
	switch (actFunct)
	{
	case SIGMOID:
		return x / (1.f + abs(x));
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
		float fx = Activation(x);
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
	if (lastSum >= 2.f)
		lastSum = 1.99999999999999999999f;
	else if (lastSum <= -2.f)
		lastSum = -1.99999999999999999999f;
	if (parent->next == NULL)
	{
		
		float targetwrtout = -(target - lastEval);
		//float outwrtnet = lastEval * (1 - lastEval);
		float outwrtnet = DerivativeActivation(lastSum);
		float nodeDelta = targetwrtout * outwrtnet;
		for (int i = 0; i < numWeights; i++)
		{
			//how impactful was weight i - following the delta rule https://en.wikipedia.org/wiki/Delta_rule
			float errorwrtweight = nodeDelta * parent->prev->nodes[i].lastEval;
			weights[i] = weights[i] - LEARNING_RATE * errorwrtweight;
		}


		//do it for bias?
		bias = bias - LEARNING_RATE * nodeDelta;

		error = nodeDelta;
		return error;
	}
	else
	{
		//calculate this nodes error
		error = 0;
		for (int i = 0; i < parent->next->numNodes; i++)//error = summation(nodeDelta * connectionWeight)
			error += parent->next->nodes[i].error * parent->next->nodes[i].weights[id];

		//float outwrtnet = lastEval * (1 - lastEval);
		float outwrtnet = DerivativeActivation(lastSum);
		float nodeDelta = error * outwrtnet;

		for (int i = 0; i < numWeights; i++)
		{
			float errorweight = nodeDelta * (parent->prev != NULL ? parent->prev->nodes[i].lastEval : parent->nn->lastInput[i]);
			weights[i] = weights[i] - LEARNING_RATE * errorweight;
		}


		//do it for bias?
		bias = bias - LEARNING_RATE * nodeDelta;

		error = nodeDelta;
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