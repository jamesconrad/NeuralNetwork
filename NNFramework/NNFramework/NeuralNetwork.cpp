#include "NeuralNetwork.h"

#include <memory> //memcpy

NeuralNetwork::NeuralNetwork(int layerCount, LayerType* layerTypes, int* nodeCounts, ActivationFunction* layerActivationFunctions, int numInputs, int* extraData)
{
	numLayers = layerCount;
	layers = (Layer*) new char[numLayers * sizeof(Layer)];
	int offset = 0;
	for (int i = 0; i < numLayers; i++)
	{
		if (extraData == nullptr)
		{
			layers[i] = Layer(nodeCounts[i], i == 0 ? numInputs : layers[i - 1].GetNumOutputs(), layerTypes[i], layerActivationFunctions[i], i, extraData);
			continue;
		}
		layers[i] = Layer(nodeCounts[i], i == 0 ? numInputs : layers[i-1].GetNumOutputs(), layerTypes[i], layerActivationFunctions[i], i, extraData + offset);
		offset += extraData[offset];
	}
}

int NeuralNetwork::FinalLayerOutputCount()
{
	return layers[numLayers - 1].GetNumOutputs();
}

void NeuralNetwork::Evaluate(float* inputs, int numInputs, float* outputValues)
{
	delete _results[0];
	delete _results[1];

	_numOutputs[0] = layers[0].GetNumOutputs();
	_results[0] = new float[_numOutputs[0]];
	_results[1] = new float[1];
	layers[0].Evaluate(inputs, numInputs, _results[0], &_numOutputs[0]);
	
	int index = 0;
	int lastindex = 0;
	for (int i = 1; i < numLayers; i++)
	{
		index = i % 2;
		delete _results[index]; //delete the one that was not done last
		_numOutputs[index] = layers[i].GetNumOutputs();
		_results[index] = new float[_numOutputs[index]];//reallocate
		layers[i].Evaluate(_results[lastindex], _numOutputs[lastindex], _results[index], &_numOutputs[index]);//store results

		lastindex = index;//store index
	}
	memcpy_s(outputValues, _numOutputs[lastindex] * sizeof(float), _results[lastindex], _numOutputs[lastindex] * sizeof(float));
	//outputValues = _results[lastindex];
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
	for (int i = 0; i < numLayers; i++)
		layers[i].LogState(runId, toFile, toConsole, stateLog);
}