#pragma once
#include <stdio.h>
#include <vector>
#define LEARNING_RATE 0.5f

class Layer;
enum LayerType;

enum ActivationFunction
{
	SIGMOID,
	TANH,
	RELU
};

class Node
{
public:
	friend Layer;

	Node(int weightCounts, Layer* parentLayer, ActivationFunction activationFunction, int id, int* extraData);

	float Evaluate(float* inputs, int numInputs);
	float Error(float target);
	float Backpropogate(float target = 0);
	void LogStructure(bool toFile, bool toConsole, FILE* file);
	void LogState(int runId, bool toFile, bool toConsole, FILE* file);

private:
	float Activation(float x);
	float DerivativeActivation(float x);

	int id;
	float bias = 0.f;
	int numWeights;
	float* weights;
	//std::vector<float> weights;
	float lastSum;
	float error;
	float lastEval = -123.f;
	Layer* parent;
	LayerType layerType;
	ActivationFunction actFunct;
};