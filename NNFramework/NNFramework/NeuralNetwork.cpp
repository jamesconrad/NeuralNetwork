#include "NeuralNetwork.h"

#include <memory> //memcpy

NeuralNetwork::NeuralNetwork(int layerCount, LayerType* layerTypes, int* nodeCounts, ActivationFunction* layerActivationFunctions, int numInputs, int* extraData)
{
	numLayers = layerCount;
	//allocate memory, avoid use of Layer constructor
	layers = (Layer**) new char[numLayers * sizeof(Layer*)];
	//layers.reserve(numLayers);
	int offset = 0;
	for (int i = 0; i < numLayers; i++)
	{
		//generate a layer and assign pointer to us
		//layers.push_back(Layer(nodeCounts[i], i == 0 ? numInputs : layers[i - 1].GetNumOutputs(), layerTypes[i], layerActivationFunctions[i], i, extraData));
		layers[i] = new Layer(nodeCounts[i], i == 0 ? numInputs : layers[i - 1]->GetNumOutputs(), layerTypes[i], layerActivationFunctions[i], i, extraData);
		layers[i]->nn = this;
	}
	
	for (int i = 0; i < numLayers; i++)
	{
		//link layers
		layers[i]->prev = i - 1 < 0 ? NULL : layers[i - 1];
		layers[i]->next = i + 1 >= numLayers ? NULL : layers[i + 1];
	}
	//allocate lastinput space
	lastInput = new float[numInputs];
}

int NeuralNetwork::FinalLayerOutputCount()
{
	return layers[numLayers - 1]->GetNumOutputs();
}

void NeuralNetwork::Evaluate(float* inputs, int numInputs, float* outputValues)
{
	//store lastinput
	memcpy_s(lastInput, sizeof(float) * numInputs, inputs, sizeof(float) * numInputs);

	//prep our memory
	_numOutputs[0] = layers[0]->GetNumOutputs();
	_results[0] = new float[_numOutputs[0]];
	_results[1] = new float[1];
	//first evaluation goes into index 0
	layers[0]->Evaluate(inputs, numInputs, _results[0], &_numOutputs[0]);
	
	//std::vector<float> r;

	int index = 0;
	int lastindex = 0;
	for (int i = 1; i < numLayers; i++)
	{
		index = i % 2; //calculate which index we are going to use
		delete _results[index]; //delete the one that was not done last
		_numOutputs[index] = layers[i]->GetNumOutputs();
		_results[index] = new float[_numOutputs[index]];//reallocate
		layers[i]->Evaluate(_results[lastindex], _numOutputs[lastindex], _results[index], &_numOutputs[index]);//store results
		//r.resize(_numOutputs[index]);
		//layers[i]->Evaluate(_results[lastindex], _numOutputs[lastindex], r.data(), &_numOutputs[index]);
		lastindex = index;//store index
	}

	//store return values
	memcpy_s(outputValues, _numOutputs[lastindex] * sizeof(float), _results[lastindex], _numOutputs[lastindex] * sizeof(float));
	delete _results[0];
	delete _results[1];
	//outputValues = _results[lastindex];
}

void NeuralNetwork::BackProp(float* actualProp)
{
	//call backprop on layers in last to first order
	layers[numLayers - 1]->Backpropogate(actualProp);
	for (int i = numLayers - 2; i >= 0; i--)
	{
		layers[i]->Backpropogate();
	}
}

bool NeuralNetwork::InitLogging(const char* stateFilePath, bool clearContents)
{
	fopen_s(&stateLog, stateFilePath, clearContents ? "w" : "a");
	if (stateLog == NULL)
	{
		printf("Unable to open state file \"%s\"", stateFilePath);
		return false;
	}
	if (clearContents) fprintf(stateLog, "logID,layerID,layerType,neuronID,activationFunction,lastValue,weights,bias\n");
	return true;
}

void NeuralNetwork::CloseLogging()
{
	fclose(stateLog);
}

void NeuralNetwork::LogState(int runId, bool toFile, bool toConsole)
{
	for (int i = 1; i < numLayers; i++)
		layers[i]->LogState(runId, toFile, toConsole, stateLog);
}