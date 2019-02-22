#ifndef DSC_CUH__
#define DSC_CUH__

#include "dsc.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>

#define CHECK(call) do { \
  const cudaError_t error = call; \
  if (error != cudaSuccess) { \
    fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
    fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
    exit(1); \
  } \
}while(0)

__device__ unsigned char atomicAddChar(unsigned char *address, unsigned char val);
__global__ void coord_UVD_to_XYZ(unsigned char *D, float *XYZ, float *P);
__global__ void coord_XYZ_to_UV(float *XYZ, int *UV, float *P);
__global__ void compute_diff_image(unsigned char* ctr, unsigned char *ref, int *UV, const unsigned char DELTA);
__global__ void recover_diff_image(unsigned char* ctr, unsigned char *ref, int *UV);
void cuda_compress(std::ifstream *files, std::ofstream *outputs, int *group, bool *is_center, int group_count, int main_stream_no, const int PIXEL_PER_THREAD, float *mat_P, int DELTA);
void cuda_recover(ImageBuffer& image_buffer, int main_stream_no, int center_no, int stream_no, const int PIXEL_PER_THREAD, float *mat_P);

#endif
