#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <fstream>
#include <iostream>


cudaError_t filterImage(cv::Mat image, cv::Mat result_image);

int kernel[9] = {
            -1, -1, -1,
            -1, 8, -1,
            -1, -1, -1 };
int kernel_total = 1, kernel_size = 3, n_blocks = 24, n_threads = 128;

__global__ void filterImagekernel(const uchar* image, const int* kernel, float kernel_total, int kernel_size, int image_width, int image_height, int image_channels, int blocks, int threads, int rows_per_block, int cols_per_thread, uchar* result_image)
{
    int initial_row, final_row, initial_col, final_col;

    //Division de filas por bloque y columnas por hilo
    initial_row = (blockIdx.x * rows_per_block) - (kernel_size / 2);
    final_row = ((blockIdx.x + 1) * rows_per_block) + (kernel_size / 2);

    initial_col = (threadIdx.x * cols_per_thread) - (kernel_size / 2);
    final_col = ((threadIdx.x + 1) * cols_per_thread) + (kernel_size / 2);

    //Validacion de valores dentro de los limites de la imagen
    if (initial_row < 0)
    {
        initial_row = 0;
    }
    if (initial_col < 0)
    {
        initial_col = 0;
    }

    if (final_row > image_height - kernel_size)
    {
        final_row = image_height - kernel_size;
    }
    if (final_col > image_width - kernel_size)
    {
        final_col = image_width - kernel_size;
    }

    //Sincronizacion de hilos
    __syncthreads();

    //Iteracion de las filas y columnas en los indices asignados previamente
    for (int row = initial_row; row < final_row; row++)
    {
        for (int col = initial_col; col < final_col; col++)
        {

            int newPixel[] = {0, 0, 0};

            // Iteracion de filas del kernel
            for (int krow = 0; krow < kernel_size; krow++)
            {
                // Iteracion de columnas del kernel
                for (int kcol = 0; kcol < kernel_size; kcol++)
                {
                    //Calculo de posisiciones del kernel y de la imagen
                    int pos = ((row + krow) * image_width * image_channels) + ((col + kcol) * image_channels);
                    int kpos = (krow * kernel_size) + kcol;

                    // Actualizacion de los nuevos valores del pixel
                    newPixel[0] += (kernel[kpos] * (int)image[pos + 1 ]);
                    newPixel[1] += (kernel[kpos] * (int)image[pos]);
                    newPixel[2] += (kernel[kpos] * (int)image[pos + 2]);
                }
            }

            //Normalizacion de datos
            for (int k = 0; k < 3; k++)
            {
                newPixel[k] = max(newPixel[k], 0);
                newPixel[k] = min(newPixel[k], 255);
            }

            //Actualizacion de valores del nuevo pixel
            result_image[(row + (kernel_size / 2)) * image_width * image_channels + (col + (kernel_size / 2)) * image_channels] = (uchar)newPixel[0];
            result_image[(row + (kernel_size / 2)) * image_width * image_channels + (col + (kernel_size / 2)) * image_channels + 1] = (uchar)newPixel[1];
            result_image[(row + (kernel_size / 2)) * image_width * image_channels + (col + (kernel_size / 2)) * image_channels + 2] = (uchar)newPixel[2];

        }
    }
}

int main(int argc, char** argv)
{
    //Validacion del numero de argumentos
    if (argc < 5)
    {
        std::cout << "Ingrese todos los argumentos necesarios para ejecutar el proceso" << std::endl;
        return -1;
    }

    //Extraccion de argumentos
    std::string path_image = argv[1];
    std::string path_save = argv[2];

    n_blocks = atoi(argv[3]);
    if (n_blocks == 0)
    {
        std::cout << "El número de bloques es invalido" << std::endl;
        return -1;
    }

    n_threads = atoi(argv[4]);
    if (n_threads == 0)
    {
        std::cout << "El número de hilos es invalido" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(path_image, cv::IMREAD_COLOR); // Read the file
    if (!image.data)
    {
        std::cout << "No se pudo abrir la imagen" << std::endl;
        return -1;
    }

    cv::Mat result_image = cv::Mat(image.rows, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));

    // Inicia el tiempo 
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL); 

    // Se agregan los vectores en paralelo.
    cudaError_t cudaStatus = filterImage(image, result_image);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
        fprintf(stderr, "Filtro de imagen fallo!");
        return 1;
    }

    // Calcular el tiempo 
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);

    // Resultados
    printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

    //Escribe los resultados en el archivo datos.txt
    std::ofstream outdata; 

    outdata.open("datos.txt",std::fstream::app); 
    if( !outdata ) { 
      std::cout<< "Error: file could not be opened" << std::endl;
    }

    char usec[25];
    sprintf(usec, "%06ld", (long int)tval_result.tv_usec);

    outdata<<"Resolucion: "<< image.rows<<std::endl;
    outdata<<"Cantidad de hilos: "<< n_threads<<std::endl;
    outdata<<"Time elapsed: "<<(long int)tval_result.tv_sec<<"."<<usec<<std::endl<<std::endl;
    outdata.close();

    //Resultados
    std::cout<<"cantidad de hilos "<< n_threads<<std::endl;
    
    std::cout<<"nombre de imagen guardada "<< path_save <<std::endl<<std::endl;

    if (!cv::imwrite(path_save, result_image)) {
        std::cout << "No se pudo guardar la imagen" << std::endl;
        return -1;
    }

    // Se debe llamar a cudaDeviceReset antes de salir para que las herramientas de creación de perfiles y seguimiento, como Nsight y Visual Profiler, muestren seguimientos completos.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t filterImage(cv::Mat image, cv::Mat result_image)
{
    //Declaracion de punteros y variables
    uchar* d_image;
    uchar* d_result_image;
    int* d_kernel;
    int image_width = image.cols;
    int image_height = image.rows;
    int image_channels = image.channels();

    cudaError_t cudaStatus = cudaSuccess;

    try
    {
        // Se escoge la GPU a utilizar, esto en caso de poseer varias GPUs.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) throw "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?";

        // Ubicacion de espacios de memoria de la GPU
        cudaStatus = cudaMalloc((void**)&d_image, image.rows * image.cols * image.channels() * sizeof(uchar));
        if (cudaStatus != cudaSuccess) throw "cudaMalloc failed! image";

        cudaStatus = cudaMalloc((void**)&d_kernel, kernel_size * kernel_size * sizeof(int));
        if (cudaStatus != cudaSuccess) throw "cudaMalloc failed! kernel";

        cudaStatus = cudaMalloc((void**)&d_result_image, image.rows * image.cols * image.channels() * sizeof(uchar));
        if (cudaStatus != cudaSuccess) throw "cudaMalloc failed! result image";


        // Copia de valores de ls vectores desde la memoria host a la memoria de la GPU.
        cudaStatus = cudaMemcpy(d_image, image.data, image.rows * image.cols * image.channels() * sizeof(uchar), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) throw "cudaMemcpy failed! image to device";

        cudaStatus = cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) throw "cudaMemcpy failed! kernel to device";


        //Calculo del numero de numero de filas y columnas que se manejaran por cada bloque e hilo
        int rows_per_block = std::ceil((float)image_height / (float)n_blocks);
        int cols_per_thread = std::ceil((float)image_width / (float)n_threads);


        // Lanzamiento del kernel en la GPU.
        filterImagekernel << <n_blocks, n_threads >> > (d_image, d_kernel, kernel_total, kernel_size, image_width, image_height, image_channels, n_blocks, n_threads, rows_per_block, cols_per_thread, d_result_image);

        // Se verifica cualquier error en el lanzamiento del kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cout << cudaGetErrorString(cudaStatus) << 0 << std::endl;
            return cudaStatus;
        }

        // Sincronizacion de CUDA que espera que el kernel finalice y retorna error en caso de obtenerlo en la ejecucion  
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            std::cout << cudaGetErrorString(cudaStatus) << 1 << std::endl;
            return cudaStatus;
        }

        // Se retorna el dato obtenido por la GPU hacia la memoria host
        cudaStatus = cudaMemcpy(result_image.data, d_result_image, image.rows * image.cols * image.channels() * sizeof(uchar), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            std::cout << cudaGetErrorString(cudaStatus) << 2 << std::endl;
            return cudaStatus;
        }
    }
    catch (char* message)
    {
        //Se libera la memoria obtenida por CUDA
        cudaFree(d_image);
        cudaFree(d_result_image);
        cudaFree(d_kernel);
        std::cout << message << std::endl;
    }

    return cudaStatus;
}

