#include "NeuralNetwork.h"

#define TRAIN_FROM_SCRATCH true
#define DATASET_FILEPATH "train-images.idx3-ubyte"
#define LABELSET_FILEPATH "train-labels.idx1-ubyte"

#define XOR false

void main()
{
	if (!XOR)
	{
		LayerType ltype[7] = { FCL, FCL, FCL, FCL, FCL, FCL, FCL };
		ActivationFunction nfunc[7] = { TANH,TANH,TANH,TANH,TANH,TANH,TANH };

		//int ncount[2] = { 784, 10 };						// 56.4%
		//int ncount[4] = { 784, 10, 50, 10 };				// 56.8%
		//int ncount[4] = { 784, 50, 50, 10 };				// 60.6%
		//int ncount[4] = { 784, 100, 50, 10 };				// 61.7%
		//int ncount[4] = { 784, 100, 100, 10 };			// 53.7%
		//int ncount[5] = { 784, 50, 50, 50, 10 };			// 64.5%
		//int ncount[6] = { 784, 50, 50, 50, 50, 10 };		// 78.3%
		//int ncount[7] = { 784, 50, 50, 50, 50, 50, 10 };	// 67.4%
		int ncount[6] = { 784, 75, 75, 75, 50, 10 };		// 84.9%
		//int ncount[6] = { 784, 125, 100, 75, 50, 10 };	// 72.9%
		//int ncount[6] = { 784, 50, 75, 75, 50, 10 };		// 77.7%
		//int ncount[6] = { 784, 75, 75, 75, 75, 10 };		// 80.2%
		//int ncount[6] = { 784, 75, 100, 75, 50, 10 };		// 82.9%
		//int ncount[6] = { 784, 100, 100, 75, 75, 10 };	// 84.4%
		//int ncount[6] = { 784, 100, 100, 100, 100, 10 };	// 83.7%

		NeuralNetwork nn(6, ltype, ncount, nfunc, 28 * 28);
		nn.ReLinkPointers();

		//load weights from file
		nn.LoadStateLog("nnstatelog.csv");


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
			int correct = 0;
			for (int i = 1; i <= 60000; i++)
			{
				//load in number image and label
				fread_s(image, 1024, 1, 784, dataset);
				fread_s(&label, 1, 1, 1, labels);
				//convert to 0-1
				for (int j = 0; j < 784; j++)
				{
					input[j] = (float)image[j] / 255.0f;
					//debug printing
					//printf("%c", input[i] > 0.66f ? 178 : input[i] > 0.33f ? 177 : input[i] > 0.0f ? 176 : ' ');
					//if (i % 28 == 0)
					//	printf("\n");
				}


				//Evaluate network using inputs
				nn.Evaluate(input, 784, output);
	
				//check output
				float max = output[0];
				char prediction = 0;
				for (int j = 0; j < 10; j++)
				{
					if (output[j] > max)
					{
						max = output[j];
						prediction = j;
					}
				}

				//log evaluation stats to console
				printf("%5i/60000 : %.4f%%  %c:%c  ", i, (float)i / 60000.0f * 100, '0' + prediction, '0' + label);
				if (label == prediction)
				{
					printf("Correct   ");
					if (i >= 60000 - 60000 / 10) correct++;//store correct ammount for pct
				}
				else
					printf("Incorrect ");

				//display final layers output
				printf("[%-+.3f, ", output[0]);
				for (int i = 1; i < 9; i++)
					printf("%-+.3f, ", output[i]);
				printf("%-+.3f]\n", output[9]);

				//create target array
				float actualResult[10] = { 0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f };
				actualResult[label] = 0.9f;

				//backpropogate
				nn.BackProp(actualResult);
				//if (i % 500 == 0)
				//	nn.LogState(i, false, true);
			}

			//output accuracy to console
			printf("Training Complete! Final 10%% accuracy %g%%\n", correct / 6000.f * 100.f);
			//log to file to load
			nn.InitLogging("nnstatelog.csv");
			nn.LogState(60000, true, false);
			nn.CloseLogging();
			printf("Exiting...\n");
		}
	}
	else
	{
		int lcount = 3;
		LayerType ltype[3] = { FCL, FCL, FCL };
		int ncount[3] = { 2, 2, 1 };
		ActivationFunction nfunc[3] = { TANH,TANH,TANH };

		NeuralNetwork nn(3, ltype, ncount, nfunc, 2);
		nn.ReLinkPointers();

		float input[2];
		bool binput[2];
		float output[1];
#define ITERATIONS 10000
		int correct = 0;
		for (int i = 1; i <= ITERATIONS; i++)
		{
			input[0] = (float)rand() / (float)RAND_MAX > 0.5 ? 0.9f : 0.1f; binput[0] = input[0] > 0.5 ? true : false;
			input[1] = (float)rand() / (float)RAND_MAX <= 0.5 ? 0.9f : 0.1f; binput[1] = input[1] > 0.5 ? true : false;
			nn.Evaluate(input, 2, output);

			float actual = binput[0] != binput[1];
			char* str = "Incorrect!";
			if (actual)
			{
				if (output[0] >= 0.5)
				{
					str = "Correct!";
					if (i >= (ITERATIONS - ITERATIONS / 10)) correct++;
				}
			}
			else
			{
				if (output[0] < 0.5)
				{
					str = "Correct!";
					if (i >= (ITERATIONS - ITERATIONS / 10)) correct++;
				}
			}

			printf("%5i/%i : %.4f%%   %s\n", i, ITERATIONS, (float)i / (float)ITERATIONS * 100, str);

			//nn.LogState(i, false, true);
			float actualResult[1] = { actual ? 0.9f : 0.1f };
			nn.BackProp(actualResult);
		}
		printf("Training Complete. Final 10%% accuracy %g%%\n", (float)correct / (ITERATIONS / 10) * 100);
		//nn.LogState(99, false, true);
		//nn.CloseLogging();*/
	}
	return;
}