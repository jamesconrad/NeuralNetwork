#include "NeuralNetwork.h"


void main()
{
	int lcount = 3;
	LayerType ltype[3] = { CONV, FCL, FCL };
	int ncount[3] = { 5,7,3 };
	ActivationFunction nfunc[3] = { TANH,SIGMOID,RELU };
	
	NeuralNetwork nn(3, ltype, ncount, nfunc, 5);

	float inputs[5] = { 0.f,1.f,0.2f,0.6f,0.1f };
	float* output = new float[nn.FinalLayerOutputCount()];
	nn.Evaluate(inputs, 5, output);

	nn.InitLogging("nnstatelog.csv");
	nn.LogState(0, true, true);
	for (int i = 1; i < 100; i++)
	{
		nn.Evaluate(output, 3, output);
		nn.LogState(i, true, false);
	}
	nn.LogState(99, false, true);
	nn.CloseLogging();
}