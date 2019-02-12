#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <omp.h>
#include <mpi.h>

unsigned char normalize(double value);
double convolution(int i, int j, unsigned char *image, int height, int width, int filterDimension, const double filter[][5]);
void saveImage(char* filename[], unsigned char* image, long fileLength);
unsigned char * readImage(char* filename[], unsigned char * image);

int main(int argc, char * argv[])
{
	int col, row;
	int width = atoi(argv[1]);
	int height = atoi(argv[2]);
	char * prefixInFile = "../../labMPI/infile";
	char * prefixOutFile = "result";
	char * fileExtension = ".bin";
	char outFileName[64], fileName[64];
	unsigned char *image;
	unsigned char *scatteredData;
	long fileSize;
	int processorsAmount, processId, scatterDataSize;
	double start, end, time;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &processorsAmount);
	MPI_Comm_rank(MPI_COMM_WORLD, &processId);

	fileSize = width * height;
	scatterDataSize = fileSize / processorsAmount;
	image = (unsigned char *)malloc(fileSize * sizeof(unsigned char));
	scatteredData = (unsigned char *)malloc(scatterDataSize * sizeof(unsigned char));

	if (processId == 0)
	{
		strcpy(fileName, prefixInFile);
		strcat(fileName, argv[1]);
		strcat(fileName, "_");
		strcat(fileName, argv[2]);
		strcat(fileName, fileExtension);
		printf("%s \n", fileName);

		strcpy(outFileName, prefixOutFile);
		strcat(outFileName, argv[1]);
		strcat(outFileName, "_");
		strcat(outFileName, argv[2]);
		strcat(outFileName, fileExtension);
		printf("%s \n", outFileName);

		image = readImage(fileName, image);
	}

	MPI_Bcast(image, fileSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	if (processId == 0)
		start = MPI_Wtime();

	const double filter[][5] = { {0,0,1,0,0},
								{0,1,2,1,0},
								{1,2,-16,2,1},
								{0,1,2,1,0},
								{0,0,1,0,0} };
#pragma omp parallel for private(col, row) schedule(dynamic, 100)
	for (int i = scatterDataSize * processId; i < scatterDataSize * (processId + 1); i++)
	{
		col = i % width;
		row = i / width;
		scatteredData[i - scatterDataSize * processId] = normalize(convolution(col, row, image, height, width, 5, filter));
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather(scatteredData, scatterDataSize, MPI_UNSIGNED_CHAR, image, scatterDataSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	if (processId == 0)
	{
		end = MPI_Wtime();
		time = (double)end - start;
		printf("Elapsed time %f \n", time);

		saveImage(outFileName, image, fileSize);
		printf("Image processing is finished\n");
	}

	free(image);
	free(scatteredData);
	MPI_Finalize();
	return 0;
}

double convolution(int i, int j, unsigned char *image, int height, int width, int filterDimension, const double filter[][5])
{
	int filterHeight, filterWidth, kernelCenter, ii, jj;
	filterHeight = filterWidth = filterDimension;
	kernelCenter = filterHeight / 2;
	double tmp = 0;
	for (long m = 0; m < filterHeight; ++m) {
		for (long n = 0; n < filterWidth; ++n) {
			ii = i + (kernelCenter - m);
			jj = j + (kernelCenter - n);
			if (ii >= 0 && ii < width && jj >= 0 && jj < height)
				tmp += image[jj * width + ii] * filter[m][n];
		}
	}
	return tmp;
}

unsigned char * readImage(char* filename[], unsigned char * image)
{
	FILE *inFile = fopen(filename, "rb");
	fseek(inFile, 0, SEEK_END);
	long long fileLength = ftell(inFile);
	fseek(inFile, 0, SEEK_SET);
	image = (unsigned char *)malloc(fileLength * sizeof(unsigned char));
	fread(image, sizeof(unsigned char), fileLength, inFile);
	fclose(inFile);
	return image;
}

void saveImage(char* filename[], unsigned char* image, long fileLength)
{
	FILE *write = fopen(filename, "wb");
	fwrite(image, sizeof(unsigned char), fileLength * sizeof(unsigned char), write);
	fclose(write);
}

unsigned char normalize(double value)
{
	if (value > 255)
		value = 255;
	else if (value < 0)
		value = 0;
	return (unsigned char)value;
}
