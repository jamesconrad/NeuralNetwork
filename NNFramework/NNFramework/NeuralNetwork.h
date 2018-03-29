#pragma once
#include <stdio.h>
#include "Layer.h"


class NeuralNetwork
{
public:
	friend Node;
	//extraData must be nullptr
	//all other pointers must be arrays of length layerCount, with each element describing that index's layer
	NeuralNetwork(int layerCount, LayerType* layerTypes, int* nodeCounts, ActivationFunction* layerActivationFunctions, int numInputs, int* extraData = nullptr);

	//note outputValues and numOutputs must be a valid empty array, it will be memcpy'd to.
	void Evaluate(float* inputs, int numInputs, float* outputValues);
	//must be called after evaluate, and re-evaluated before backproping again
	void BackProp(float* actualResult);
	int FinalLayerOutputCount();
	//returns true if file opened succesfully, prints to console error info
	bool InitLogging(const char* stateFilePath, bool clearContents = true);
	void CloseLogging();
	//Initlogging must have been called if toFile == true
	void LogState(int runId, bool toFile, bool toConsole);

	bool LoadStateLog(char* stateFilePath);

	void ReLinkPointers()
	{
		for (int i = 0; i < numLayers; i++)
		{
			layers[i]->ReLink();
			layers[i]->prev = i - 1 < 0 ? NULL : layers[i - 1];
			layers[i]->next = i + 1 >= numLayers ? NULL : layers[i + 1];
		}
	};
private:
	//void LogStructure(bool toFile, bool toConsole);
	FILE* structLog;
	FILE* stateLog;
	int numLayers;
	Layer** layers;
	//std::vector<Layer> layers;
	float* lastInput;
	//for evaluation calculations and returns
	int _numOutputs[2];
	float* _results[2];

};