#pragma once
#include <stdio.h>
#include "Layer.h"


class NeuralNetwork
{
public:
	//extraData must be in format [length, data1, data2..., length, data1, data2] where every length indicates the next layer
	NeuralNetwork(int layerCount, LayerType* layerTypes, int* nodeCounts, ActivationFunction* layerActivationFunctions, int numInputs, int* extraData = nullptr);
	
	//note outputValues and numOutputs must be an empty array, it will be memcpy'd to.
	void Evaluate(float* inputs, int numInputs, float* outputValues);
	int FinalLayerOutputCount();
	//returns true if file opened succesfully, prints to console error info
	bool InitLogging(const char* stateFilePath, bool clearContents = true);
	void CloseLogging();
	void LogState(int runId, bool toFile, bool toConsole);
private:
	//void LogStructure(bool toFile, bool toConsole);
	FILE* structLog;
	FILE* stateLog;
	int numLayers;
	Layer* layers;

	//for evaluation calculations and returns
	int _numOutputs[2];
	float* _results[2];

};