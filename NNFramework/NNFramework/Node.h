#pragma once
#include <stdio.h>

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
	//cwidth offset in each direction from center, stride is current position
	Node(int weightCounts, Layer* parentLayer, ActivationFunction activationFunction, int id, int* extraData);

	float Evaluate(float* inputs, int numInputs);

	void LogStructure(bool toFile, bool toConsole, FILE* file);
	void LogState(int runId, bool toFile, bool toConsole, FILE* file);

private:
	float Activation(float x);
	float DerivativeActivation(float x);

	int id;
	float bias = 0.f;
	int numWeights;
	int cWidth;
	int cStride;
	int inputWidth;
	float* weights;
	int* links;
	float lastEval = -123.f;
	Layer* parent;
	LayerType layerType;
	ActivationFunction actFunct;
};

class Filter
{
public:
	Filter(int _kernels, int _stride, int _width, int _filters, int _inputWidth, int _inputDepth)
	{
		kernels = _kernels;
		filters = _filters;
		stride = _stride;
		width = _width;
		inputWidth = _inputWidth;
		resolution = width * width;
		kernelSize = filters * resolution;
		numWeights = kernels * filters * resolution;
		//for each kernel, create a set of filters containing a set of weights
		weights = new float[numWeights];
		for (int i = 0; i < numWeights; i++)
		{
			weights[i] = 0;
		}
	};

	void Evaluate(float* input, float* output, int inResolution, int* outResolution, int inDepth, int* outDepth)
	{

		for (int k = 0; k < kernels; k++)
		{
			for (int f = 0; f < filters; f++)
			{

			}
		}
	}

private:

	int Index(int x, int y, int _width, int kernel, int filter) { return (x % _width) + (y * _width) + kernel * kernelSize + filter * resolution; }

	int kernelSize;
	int numWeights;
	int kernels;
	int filters;
	int stride;
	int width;
	int inputWidth;
	int resolution;
	float* eval;
	//kernel, filter, weight
	float* weights;
};