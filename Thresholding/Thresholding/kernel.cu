#include <iostream>
#include "lodepng.h"
#include "lodepng.c"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void color_balancing(unsigned char * input,int* min, int* max)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = 0; i < 3; ++i)
	{
		float temp = 255 / (float)(max[i] - min[i]);
		input[id * 3 + i] = (input[id * 3 + i]<min[i]) ? min[i] : (input[id * 3 + i]>max[i]) ? max[i] : input[id * 3 + i];
		input[id * 3 + i] = (int)((input[id * 3 + i] - min[i]) *temp);
	}


}
__global__ void hist_equalisation(unsigned char * input, int* eq_hist1, int* eq_hist2, int* eq_hist3)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
		input[id * 3 ] = eq_hist1[input[id * 3]];
		input[id * 3+1] = eq_hist2[input[id * 3+1]];
		input[id * 3+2] = eq_hist3[input[id * 3+2]];

}
__global__ void kernelapplyFilter(unsigned char * image, unsigned char * output_image, float* filter, int filterDim, int imageInH, int imageInW)
{	
	float sumR = 0, sumG = 0, sumB = 0;
	int idT = blockIdx.x * blockDim.x + threadIdx.x;
	if (idT > imageInW && idT < (imageInH - 1)*imageInW)
	{
		int i = idT / (imageInW);
		int j = idT % (imageInW);
		int k = filterDim / 2;
		
		for (int fi = -k; fi <= k; fi++)
		{
			for (int fj = -k; fj <= k; fj++)
			{
				sumR += filter[(fi + k)*filterDim + fj + k] * image[((i + fi)*(imageInW)+j + fj) * 3];
				sumG += filter[(fi + k)*filterDim + fj + k] * image[((i + fi)*(imageInW)+j + fj) * 3 + 1];
				sumB += filter[(fi + k)*filterDim + fj + k] * image[((i + fi)*(imageInW)+j + fj) * 3 + 2];
			}
		}

		output_image[(i*(imageInW)+j) * 3] = (int) ((sumR > 255) ? 255 : sumR);
		output_image[(i*(imageInW)+j) * 3 + 1] = (int) ((sumG > 255) ? 255 : sumG);
		output_image[(i*(imageInW)+j) * 3 + 2] = (int) ((sumB > 255) ? 255 : sumB);
	}
}
__global__ void kernelNegative(unsigned char * image)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	image[id * 3] = ~image[id * 3];
	image[id * 3 + 1] = ~image[id * 3 + 1];
	image[id * 3 + 2] = ~image[id * 3 + 2];
}
__global__ void kernelLogartithmic(unsigned char * image, double constant)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idB = id * 3;
	int r = constant * (__logf(image[idB]) + 1);
	int g = constant * (__logf(image[idB + 1]) + 1);
	int b = constant * (__logf(image[idB + 2]) + 1);
	image[idB] = (r>255) ? 255 : r;
	image[idB + 1] = (g>255) ? 255 : g;
	image[idB + 2] = (b>255) ? 255 : b;
}
__global__ void kernelPowLow(unsigned char * image, double constant, double gamma)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idB = id * 3;
	int r = constant * (powf(image[idB], gamma));
	int g = constant * (powf(image[idB + 1], gamma));
	int b = constant * (powf(image[idB + 2], gamma));
	image[idB] = (r > 255) ? 255 : r;
	image[idB + 1] = (g > 255) ? 255 : g;
	image[idB + 2] = (b > 255) ? 255 : b;
}
__global__ void kernelPiecewise(unsigned char* image, int down, int up)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idB = id * 3;
	int sum = (image[idB] * 0.3 + image[idB + 1] * 0.59 + image[idB + 2] * 0.11)/2.55;
	if (sum < down || sum > up)
	{
		image[idB] = 80;
		image[idB + 1] = 80;
		image[idB + 2] = 80;
	}
	else
	{
		image[idB] = 160;
		image[idB + 1] = 160;
		image[idB + 2] = 160;
	}

}


int main(int argc, char ** argv)
{


	int blockSize = 256;
	int gridSize;
	unsigned char * array;
	int *min, *max;
	size_t pngsize;
	 unsigned char *png;
	float *d_filter;
	const char * filename = "Lenna_test.png";
	lodepng_load_file(&png, &pngsize, filename);

	unsigned char *image,*output_image;
	unsigned int width, height;
	//ucitavanje slike
	unsigned int error = lodepng_decode24(&image, &width, &height, png, pngsize);


	if (error != 0){
		std::cout << "error " << error << ": " << lodepng_error_text(error) << std::endl;
	}
	//ukupan broj piksela u slici
	unsigned int N = width*height;

	//histogrami: pocetni, kumulativni, ujednaceni
	 int hist[3][255];
	 int cumul_hist[3][255];
	 int equal_hist[3][255];

	for (int i = 0; i < 255; i++)
	{
		hist[0][i] = 0;
		cumul_hist[0][i] = 0;
		hist[1][i] = 0;
		cumul_hist[1][i] = 0;
		hist[2][i] = 0;
		cumul_hist[2][i] = 0;
	}
	//formiranje sva tri histograma
	for (int i = 0; i < N; i++)
	{
		hist[0][image[3 * i]]++;
		hist[1][image[3 * i+1]]++;
		hist[2][image[3 * i+2]]++;
	}

	cumul_hist[0][0] = hist[0][0];
	cumul_hist[1][0] = hist[1][0];
	cumul_hist[2][0] = hist[2][0];
	for (int i = 0; i < 3; i++)
	{
		for (int j = 1; j < 255; j++)
		{
			cumul_hist[i][j] = cumul_hist[i][j - 1] + hist[i][j];
		}
	}
	float temp[3];
	 temp[0] = 255.0 / cumul_hist[0][254];
	 temp[1] = 255.0 / cumul_hist[1][254];
	 temp[2] = 255.0 / cumul_hist[2][254];
	 for (int i = 0; i < 3; i++)
	 {
		 for (int j = 0; j < 255; j++)
		 {
			 equal_hist[i][j] = cumul_hist[i][j] * temp[i]+0.5;
		 }
	 }
	 
	 int operation = 0;
	 printf(" Filteri:\n");
	 printf(" 1: Balansiranje boje \n 2: Ujednacavanje histograma \n 3: Konvolucija \n 4: Negativ \n 5: Logaritamska transformacija \n 6: Gama transformacija \n 7: Naglasavanje dela \n 8: Blurovanje slike \n");
	 printf("---------------------------\n");
	 printf(" Unesite broj zeljenog filtera: ");

	 scanf("%d", &operation);
	
	 gridSize = N / blockSize;
	 switch (operation)
	 {
		 //Color balancing
		 case 1:
		 {
				   int s1, s2;
				   printf("\n Unesite procente za odsecanje: \n");
				   //printf(" Donja granica: ");
				   scanf("%d ", &s1);
				   //printf("\n");
				   //printf(" Gornja granica: ");
				   //fflush(stdin);
				   scanf("%d", &s2);
				   //printf("\n");

				   int vmin[3] = { 0, 0, 0 };
				   int vmax[3] = { 254, 254, 254 };
				   float temp_min = N* ((float)s1 / 100);
				   float  temp_max = N*((float)1 - (float)s2 / 100);
				   for (int i = 0; i < 3; i++)
				   {
					   while (cumul_hist[i][vmin[i] + 1] <= temp_min)
						   vmin[i]++;
					   while (cumul_hist[i][vmax[i] - 1] > temp_max)
						   vmax[i]--;
					   if (vmax[i] < 255 - 1)
						   vmax[i]++;
				   }
				   cudaMalloc((void **)& array, sizeof (char)* width*height * 3);
				   cudaMemcpy(array, image, sizeof (char)* width*height * 3, cudaMemcpyHostToDevice);
				   cudaMalloc((void **)& max, sizeof (int)* 3);
				   cudaMemcpy(max, vmax, sizeof (int)* 3, cudaMemcpyHostToDevice);
				   cudaMalloc((void **)& min, sizeof (int)* 3);
				   cudaMemcpy(min, vmin, sizeof (int)* 3, cudaMemcpyHostToDevice);
				   color_balancing << <gridSize, blockSize >> > (array, min, max);
				   cudaMemcpy(image, array, sizeof (char)* width*height * 3, cudaMemcpyDeviceToHost);

				   cudaFree(array);
				   cudaFree(min);
				   cudaFree(max);
		 }
			 break;
		 //Ujednacavanje histograma
		 case 2:
		 {
				   int * eq_hist1, *eq_hist2, *eq_hist3;
				   cudaMalloc((void **)& array, sizeof (char)* width*height * 3);
				   cudaMemcpy(array, image, sizeof (char)* width*height * 3, cudaMemcpyHostToDevice);
				
				   cudaMalloc((void **)& eq_hist1, sizeof (int)* 255*3);
				   cudaMemcpy(eq_hist1, equal_hist[0], sizeof (int)*255, cudaMemcpyHostToDevice);
				   
				   cudaMalloc((void **)& eq_hist2, sizeof (int)* 255 * 3);
				   cudaMemcpy(eq_hist2, equal_hist[1], sizeof (int)* 255, cudaMemcpyHostToDevice);
				   cudaMalloc((void **)& eq_hist3, sizeof (int)* 255 * 3);
				   cudaMemcpy(eq_hist3, equal_hist[2], sizeof (int)* 255, cudaMemcpyHostToDevice);
				   hist_equalisation << <gridSize, blockSize >> > (array, eq_hist1, eq_hist2, eq_hist3);
				   cudaMemcpy(image, array, sizeof (char)* width*height * 3, cudaMemcpyDeviceToHost);

				   cudaFree(array);
				   cudaFree(eq_hist1);
				   cudaFree(eq_hist2);
				   cudaFree(eq_hist3);
		 }
			 break;
		//Konvolucija
		 case 3:
		 {		
				   int filter_num;
				   printf(" Konvolucioni kerneli:\n");
				   printf(" 1: Naglasavanje ivica \n 2: Naglasavanje ivica 2 \n 3: Laplasov kernel \n 4: Izostravanje \n 5: Emboss \n 6: Prosecna vrednost \n");
				   printf(" Unesite broj zeljenog konvolucionog kernela: ");
				   scanf("%d", &filter_num);
				   float filter1[9] = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
				   float filter2[9] = { 1, 0, -1, 0, 0, 0, -1, 0, 1 };
				   float filter3[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 }; // laplacian filter
				   float filter4[9] = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
				   float filter5[9] = { -2, -1, 0, -1, 1, 1, 0, 1, 2 };//emboss
				   float filter6[9] = { 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f };//low pass filter
				   float* filter;
				   switch (filter_num)
				   {
					   case 1:
					   {
								 filter = filter1;
					   }
						   break;
					   case 2:
					   {
								 filter = filter2;
					   }
						   break;
					   case 3:
					   {
								 filter = filter3;
					   }
						   break;
					   case 4:
					   {
								 filter = filter4;
					   }
						   break;
					   case 5:
					   {
								 filter = filter5;
					   }
						   break;
					   case 6:
					   {
								 filter = filter6;
					   }
						   break;

					   default:
							filter = filter1;
						   break;
				   }

				   int filterDim = 3;
				   
				   cudaMalloc((void **)& array, sizeof (char)* height*width* 3);
				   cudaMemcpy(array, image, sizeof (char)* height*width * 3, cudaMemcpyHostToDevice);
				   cudaMalloc((void **)& output_image, sizeof (char)*width*height * 3);
				   
				 
				   cudaMalloc((void **)&d_filter, sizeof (float)* 9);
				   cudaMemcpy(d_filter, filter, sizeof (float)* 9, cudaMemcpyHostToDevice);
				  
				   kernelapplyFilter << <gridSize, blockSize >> > (array, output_image, d_filter, filterDim,height,width);
				   cudaMemcpy(image, output_image, sizeof (char)* width*height * 3, cudaMemcpyDeviceToHost);

				   cudaFree(array);
				   cudaFree(output_image);
				   cudaFree(d_filter);

		 }
			 break;
		//Negativ
		 case 4:
		 {
				   cudaMalloc((void **)& array, sizeof (char)* height*width * 3);
				   cudaMemcpy(array, image, sizeof (char)* height*width * 3, cudaMemcpyHostToDevice);
				   kernelNegative << <gridSize, blockSize >> >(array);
				   cudaMemcpy(image, array, sizeof (char)* width*height * 3, cudaMemcpyDeviceToHost);
				   cudaFree(array);
		 } 
			 break;
		//Logaritamska transformacija
		 case 5:
			 {		
				   double constant = 0;
				   printf("%s", "Unesite zeljenu vrednost konstante za logaritamsku transformaciju: ");
				   scanf("%lf", &constant);
				   //printf("%f\n", constant);
				   cudaMalloc((void **)& array, sizeof (char)* height*width * 3);
				   cudaMemcpy(array, image, sizeof (char)* height*width * 3, cudaMemcpyHostToDevice);
				   kernelLogartithmic << <gridSize, blockSize >> >(array, constant);
				   cudaMemcpy(image, array, sizeof (char)* width*height * 3, cudaMemcpyDeviceToHost);
				   cudaFree(array);
			 }
			 break;
		//Gama transformacija
		 case 6:
		 {
				   double constant = 0, gama = 0;;
				   printf("%s\n", "Unesite zeljenu vrednost konstante za ovu transformaciju: ");
				   scanf("%lf", &constant);
				   printf("%s\n", "Unesite zeljenu vrednost gama za ovu transformaciju: ");
				   scanf("%lf", &gama);
				   cudaMalloc((void **)& array, sizeof (char)* height*width * 3);
				   cudaMemcpy(array, image, sizeof (char)* height*width * 3, cudaMemcpyHostToDevice);
				   kernelPowLow << <gridSize, blockSize >> >(array, constant,gama);
				   cudaMemcpy(image, array, sizeof (char)* width*height * 3, cudaMemcpyDeviceToHost);
				   cudaFree(array);
		 }
			 break;
		//Naglasavanje dela
		 case 7:
		 {
				   
					   int down = 0, up = 0;;
					   printf("%s", "Unesite zeljenu vrednost donje granice osvetljenosti u procentima: ");
					   scanf("%d", &down);
					   printf("%s", "Unesite zeljenu vrednost gornje granice osvetljenosti u procentima: ");
					   scanf("%d", &up);
					   cudaMalloc((void **)& array, sizeof (char)* height*width * 3);
					   cudaMemcpy(array, image, sizeof (char)* height*width * 3, cudaMemcpyHostToDevice);
					   kernelPiecewise << <gridSize, blockSize >> >(array, down, up);
					   cudaMemcpy(image, array, sizeof (char)* width*height * 3, cudaMemcpyDeviceToHost);
					   cudaFree(array);
				 
		 }
			 break;
		// Blurovanje slike
		 case 8:
		 {
				   int count;
				   printf("%s", "Unesite koliko puta zelite da primenite blur efekat na sliku: ");
				   scanf("%d", &count);

				   float filter[9] = { 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f };//low pass filter
				   int filterDim = 3;

				   cudaMalloc((void **)& array, sizeof (char)* height*width * 3);
				   cudaMalloc((void **)& output_image, sizeof (char)*width*height * 3);
				   cudaMalloc((void **)&d_filter, sizeof (float)* 9);

				   for (int i = 0; i < count; i++)
				   {
					   cudaMemcpy(array, image, sizeof (char)* height*width * 3, cudaMemcpyHostToDevice);
					   cudaMemcpy(d_filter, filter, sizeof (float)* 9, cudaMemcpyHostToDevice);
					   kernelapplyFilter << <gridSize, blockSize >> > (array, output_image, d_filter, filterDim, height, width);
					   cudaMemcpy(image, output_image, sizeof (char)* width*height * 3, cudaMemcpyDeviceToHost);
				   }

				   cudaFree(array);
				   cudaFree(output_image);
				   cudaFree(d_filter);

		 }
			 break;
	 }
	 cudaError_t errSync = cudaGetLastError();
	 cudaError_t errAsync = cudaDeviceSynchronize();
	 if (errSync != cudaSuccess)
		 printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	 if (errAsync != cudaSuccess)
		 printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
	 cudaFree(array);
	lodepng_encode24_file("lenna_processed.png", image, width, height);

	return 0;
} 

