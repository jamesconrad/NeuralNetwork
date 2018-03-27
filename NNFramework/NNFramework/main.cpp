#include "NeuralNetwork.h"

#define TRAIN_FROM_SCRATCH true
#define DATASET_FILEPATH "train-images.idx3-ubyte"
#define LABELSET_FILEPATH "train-labels.idx1-ubyte"


void main()
{
	int lcount = 6;
	LayerType ltype[6] = { FCL, FCL, FCL, FCL, FCL, FCL };
	//int ncount[6] = { 28*28,14*14*6,100*16,120,84,10};
	int ncount[6] = { 28 * 28, 10, 10, 10, 10, 10 };
	ActivationFunction nfunc[6] = { SIGMOID,SIGMOID,SIGMOID,SIGMOID,SIGMOID,SIGMOID };
	
	NeuralNetwork nn(6, ltype, ncount, nfunc, 28*28);
	//nn.InitLogging("nnstatelog.csv");
	//nn.LogState(0, true, false);

	if (TRAIN_FROM_SCRATCH)
	{
		FILE* dataset;
		FILE* labels;
		fopen_s(&dataset, DATASET_FILEPATH, "rb");
		fopen_s(&labels, LABELSET_FILEPATH, "rb");

		int resolution = 784;//we know the input is 28x28 and 60000 images

		unsigned char image[1024];
		float input[784];
		float output[10];
		unsigned char label;
		fread_s(image, 784, sizeof(int), 4, dataset);	//set start points to file data
		fread_s(image, 784, sizeof(int), 2, labels);	//set start points to file data

		for (int i = 1; i < 60000; i++)
		{
			fread_s(image, 1024, 1, 784, dataset);
			fread_s(&label, 1, 1, 1, labels);
			for (int i = 0; i < 784; i++)
			{
				input[i] = (float)image[i] / 255.0f;
				//printf("%c", input[i] > 0.66f ? 178 : input[i] > 0.33f ? 177 : input[i] > 0.0f ? 176 : ' ');
				//if (i % 28 == 0)
				//	printf("\n");
			}

			nn.Evaluate(input, 784, output);
			//nn.LogState(1, true, false);
			//nn.CloseLogging();
			float max = output[0];
			char prediction = 0;
			for (int i = 1; i < 10; i++)
			{
				if (output[i] > max)
				{
					max = output[i];
					prediction = i;
				}
			}

			printf("%i/60000 : %g%%\t\t%c:%c\t", i, (float)i / 60000.0f * 100, '0' + prediction, '0' + label);
			if (label == prediction)
			{
				printf("Correct!\n");
			}
			else
				printf("Incorrect!\n");

			printf("[%.8f, ", output[0]);
			for (int i = 1; i < 9; i++)
				printf("%.8f, ", output[i]);
			printf("%.8f]\n", output[9]);

			nn.LogState(i, false, true);

			float actualResult[10] = {0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f};
			actualResult[label] = 0.9f;

			nn.BackProp(actualResult);
			//if (i % 100 == 0)
			//	nn.LogState(i, false, true);
		}

	}

	//float inputs[5] = { 0.f,1.f,0.2f,0.6f,0.1f };
	//float* output = new float[nn.FinalLayerOutputCount()];
	//nn.Evaluate(inputs, 5, output);
	//
	//nn.InitLogging("nnstatelog.csv");
	////nn.LogState(0, true, true);
	//for (int i = 1; i < 100; i++)
	//{
	//	nn.Evaluate(output, 3, output);
	//	//nn.LogState(i, true, false);
	//}
	////nn.LogState(99, false, true);
	//nn.CloseLogging();
}