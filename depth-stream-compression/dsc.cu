#include "dsc.cuh"
#include "dsc.h"
#include <fstream>

using namespace std;

__device__ unsigned char atomicAddChar(unsigned char* address, unsigned char val) {
  // TODO: big endian & small endian
  unsigned int *base_address = (unsigned int *)(address - ((size_t)address & 3));
  int shift = ((size_t)address & 3) * 8;
  unsigned int long_val = ((unsigned int)val) << shift;
  unsigned int long_old = atomicAdd(base_address, long_val);

  if (shift != 24) {
    unsigned int overflow = ((long_old & (0xff << shift)) + long_val) & (0xff << shift + 8);
    if (overflow) atomicSub(base_address, overflow);
  }
  return (unsigned char)((long_old & (0xff << shift)) >> shift);
}

__global__ void coord_UVD_to_XYZ(unsigned char *D, float *XYZ, float *P) {
  int U = threadIdx.x, V = blockIdx.x;

  // increase pointers to corresponding location
  D += V * IMG_WIDTH + U;
  XYZ += (V * IMG_WIDTH + U) * 3;

  float X = 0, Y = 0, Z = 0;

  Z = 1.0f / ((D[0] / 255.0f) * (1.0f / MinZ - 1.0f / MaxZ) + 1.0f / MaxZ);

  float c0, c1, c2;
  c0 = Z * MAT_Pn(P, 0, 2) + MAT_Pn(P, 0, 3);
  c1 = Z * MAT_Pn(P, 1, 2) + MAT_Pn(P, 1, 3);
  c2 = Z * MAT_Pn(P, 2, 2) + MAT_Pn(P, 2, 3);

  V = IMG_HEIGHT - V - 1;

  Y = U * (c1 * MAT_Pn(P, 2, 0) - MAT_Pn(P, 1, 0) * c2)
    + V * (c2 * MAT_Pn(P, 0, 0) - MAT_Pn(P, 2, 0) * c0)
    + (c0 * MAT_Pn(P, 1, 0) - MAT_Pn(P, 0, 0) * c1);

  Y /= V * (MAT_Pn(P, 2, 0) * MAT_Pn(P, 0, 1) - MAT_Pn(P, 2, 1) * MAT_Pn(P, 0, 0))
    + U * (MAT_Pn(P, 1, 0) * MAT_Pn(P, 2, 1) - MAT_Pn(P, 1, 1) * MAT_Pn(P, 2, 0))
    + (MAT_Pn(P, 0, 0) * MAT_Pn(P, 1, 1) - MAT_Pn(P, 1, 0) * MAT_Pn(P, 0, 1));

  X = Y * (MAT_Pn(P, 0, 1) - MAT_Pn(P, 2, 1) * U) + c0 - c2 * U;
  X /= MAT_Pn(P, 2, 0) * U - MAT_Pn(P, 0, 0);

  // save coordinate XYZ
  XYZ[0] = X; XYZ[1] = Y; XYZ[2] = Z;
}

__global__ void coord_XYZ_to_UV(float* XYZ, int* UV, float* P) {
  XYZ += (blockIdx.x * blockDim.x + threadIdx.x) * 3;
  UV += (blockIdx.x * blockDim.x + threadIdx.x) * 2;

  float X = XYZ[0], Y = XYZ[1], Z = XYZ[2];
  float U, V, W;
  U = MAT_P(P, 0, 0, 0) * X + MAT_P(P, 0, 0, 1) * Y + MAT_P(P, 0, 0, 2) * Z + MAT_P(P, 0, 0, 3);
  V = MAT_P(P, 0, 1, 0) * X + MAT_P(P, 0, 1, 1) * Y + MAT_P(P, 0, 1, 2) * Z + MAT_P(P, 0, 1, 3);
  W = MAT_P(P, 0, 2, 0) * X + MAT_P(P, 0, 2, 1) * Y + MAT_P(P, 0, 2, 2) * Z + MAT_P(P, 0, 2, 3);
  U /= W;
  V /= W;

  V = IMG_HEIGHT - V - 1;

  UV[0] = (int)__float2int_rd(U + 0.5f);
  UV[1] = (int)__float2int_rd(V + 0.5f);
}

__global__ void compute_diff_image(unsigned char* ctr, unsigned char *ref, int *UV, const unsigned char DELTA) {
  // 8 pixels per thread
  const int PIXEL_PER_THREAD = 8;
  int idx = blockIdx.x * IMG_WIDTH + threadIdx.x * PIXEL_PER_THREAD;
  UV += idx * 2;
  ctr += idx;

  int R = 0;
  int G = R + IMG_SIZE;
  int B = G + IMG_SIZE;
  int D = B + IMG_SIZE;

  unsigned char diff_R, diff_G, diff_B, diff_D;

  int U, V, ref_idx;
  for (int i = 0; i < PIXEL_PER_THREAD; i++) {
    U = UV[i * 2 + 0]; V = UV[i * 2 + 1];

    if (U >= 0 && V >= 0 && U < IMG_WIDTH && V < IMG_HEIGHT) {

      ref_idx = V * IMG_WIDTH + U;

      diff_R = /* ref[ref_idx + R] */ -ctr[i + R];
      diff_G = /* ref[ref_idx + G] */ -ctr[i + G];
      diff_B = /* ref[ref_idx + B] */ -ctr[i + B];
      diff_D =                        -ctr[i + D];

      // if(diff_R < DELTA || diff_R > 255 - DELTA) { diff_R = -ref[ref_idx + R]; } else { diff_R = -ctr[i + R]; } 
      // if(diff_G < DELTA || diff_G > 255 - DELTA) { diff_G = -ref[ref_idx + G]; } else { diff_R = -ctr[i + R]; } 
      // if(diff_B < DELTA || diff_B > 255 - DELTA) { diff_B = -ref[ref_idx + B]; } else { diff_R = -ctr[i + R]; } 

      atomicAddChar(&ref[ref_idx + R], diff_R);
      atomicAddChar(&ref[ref_idx + G], diff_G);
      atomicAddChar(&ref[ref_idx + B], diff_B);
      atomicAddChar(&ref[ref_idx + D], diff_D);
    }
  }
}

__global__ void recover_diff_image(unsigned char* ctr, unsigned char *ref, int *UV) {
  // 8 pixels per thread
  const int PIXEL_PER_THREAD = 8;
  int idx = blockIdx.x * IMG_WIDTH + threadIdx.x * PIXEL_PER_THREAD;
  UV += idx * 2;
  ctr += idx;

  int R = 0;
  int G = R + IMG_SIZE;
  int B = G + IMG_SIZE;
  int D = B + IMG_SIZE;

  //unsigned char diff_R, diff_G, diff_B, diff_D;

  int U, V, ref_idx;
  for (int i = 0; i < PIXEL_PER_THREAD; i++) {
    U = UV[i * 2 + 0]; V = UV[i * 2 + 1];

    if (U >= 0 && V >= 0 && U < IMG_WIDTH && V < IMG_HEIGHT) {

      ref_idx = V * IMG_WIDTH + U;

      atomicAddChar(&ref[ref_idx + R], ctr[i + R]);
      atomicAddChar(&ref[ref_idx + G], ctr[i + G]);
      atomicAddChar(&ref[ref_idx + B], ctr[i + B]);
      atomicAddChar(&ref[ref_idx + D], ctr[i + D]);
    }
  }
}

void cuda_compress(ifstream *files, ofstream *outputs, int *group, bool *is_center, int group_count, int main_stream_no, const int PIXEL_PER_THREAD, float *mat_P, int DELTA) {
  char frame_rate_str[64];
  char front_color[3] = { 0, 0, 0 };
  char back_color[3] = { 255, 255, 255 };
  CImgDisplay disp;

  const int IMG_RGB_SIZE = IMG_SIZE * IMG_CHANNEL;
  const int IMG_RGBD_SIZE = IMG_SIZE * (IMG_CHANNEL + 1);
  const int IMG_BUFFER_SIZE = IMG_RGBD_SIZE * TOTAL_CAMS;
  unsigned char *diff_img_buffer = (unsigned char *)malloc(IMG_BUFFER_SIZE);
  unsigned char *img_buffer = (unsigned char *)malloc(IMG_BUFFER_SIZE);

  unsigned char *bitmap = (unsigned char*)malloc(IMG_SIZE / 2);
  int *frame_start_pos = (int*)malloc(TOTAL_FRAMES * TOTAL_CAMS * sizeof(int) * 2);

  unsigned char *d_img_buffer, *d_diff_img_buffer;
  float *d_P, *d_XYZ;
  int *d_UV;
  CHECK(cudaMalloc(&d_img_buffer, IMG_BUFFER_SIZE));
  CHECK(cudaMalloc(&d_diff_img_buffer, IMG_BUFFER_SIZE));
  CHECK(cudaMalloc(&d_P, MAT_P_SIZE * TOTAL_CAMS * sizeof(float)));
  CHECK(cudaMalloc(&d_XYZ, IMG_SIZE * 3 * sizeof(float)));
  CHECK(cudaMalloc(&d_UV, IMG_SIZE * 2 * sizeof(int)));
  CHECK(cudaMemcpy(d_P, mat_P, MAT_P_SIZE * TOTAL_CAMS * sizeof(float), cudaMemcpyHostToDevice));

  int frame_start_pos_buffer[TOTAL_CAMS] = { 0 };
  for (int frame_no = 0; frame_no < TOTAL_FRAMES; frame_no++) {
    // ----- read images to gpu ----- //
    for (int i = 0; i < TOTAL_CAMS; i++) {
      files[i].seekg(frame_no * IMG_RGBD_SIZE);
      files[i].read((char*)img_buffer + i * IMG_RGBD_SIZE, IMG_RGBD_SIZE);
    }
    CHECK(cudaMemcpy(d_img_buffer, img_buffer, IMG_BUFFER_SIZE, cudaMemcpyHostToDevice));

    // ----- group based compress ----- //
    for (int group_no = 0; group_no < group_count; group_no++) {
      // ----- reference stream compress ----- //

      // find center stream no
      int center_no = main_stream_no;
      for (int i = 0; i < TOTAL_CAMS; i++) {
        if (group[i] == group_no && is_center[i]) {
          center_no = i;
          break;
        }
      }

      coord_UVD_to_XYZ << <IMG_HEIGHT, IMG_WIDTH >> > (d_img_buffer + IMG_RGBD_SIZE * center_no + IMG_RGB_SIZE, d_XYZ, GET_MAT_P(d_P, center_no));
      CHECK(cudaDeviceSynchronize());
      // compress
      for (int stream_no = 0; stream_no < TOTAL_CAMS; stream_no++) {
        // not compress main stream
        if (stream_no == main_stream_no) continue;
        // skip center stream itself
        if (is_center[stream_no]) continue;
        if (group[stream_no] != group_no) continue;

        coord_XYZ_to_UV << <IMG_HEIGHT, IMG_WIDTH >> > (d_XYZ, d_UV, GET_MAT_P(d_P, stream_no));
        CHECK(cudaDeviceSynchronize());
        compute_diff_image << <IMG_HEIGHT, IMG_WIDTH / PIXEL_PER_THREAD >> > (d_img_buffer + IMG_RGBD_SIZE * center_no, d_img_buffer + IMG_RGBD_SIZE * stream_no, d_UV, DELTA);
        CHECK(cudaDeviceSynchronize());
      }

      // ----- center stream compress ----- //
      if (center_no == main_stream_no) continue;
      coord_UVD_to_XYZ << <IMG_HEIGHT, IMG_WIDTH >> > (d_img_buffer + IMG_RGBD_SIZE * main_stream_no + IMG_RGB_SIZE, d_XYZ, GET_MAT_P(d_P, main_stream_no));
      CHECK(cudaDeviceSynchronize());
      coord_XYZ_to_UV << <IMG_HEIGHT, IMG_WIDTH >> > (d_XYZ, d_UV, GET_MAT_P(d_P, center_no));
      CHECK(cudaDeviceSynchronize());
      compute_diff_image << <IMG_HEIGHT, IMG_WIDTH / PIXEL_PER_THREAD >> > (d_img_buffer + IMG_RGBD_SIZE * main_stream_no, d_img_buffer + IMG_RGBD_SIZE * center_no, d_UV, DELTA);
      CHECK(cudaDeviceSynchronize());
    }

    CHECK(cudaMemcpy(diff_img_buffer, d_img_buffer, IMG_BUFFER_SIZE, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    // display
    for (int i = 0; i < TOTAL_CAMS; i++) {
      unsigned char *diff_img = diff_img_buffer + i * IMG_RGBD_SIZE;
      int count = 0;
      for (int i = 0; i < IMG_HEIGHT * 4; i++) {
        unsigned char *row = diff_img + IMG_WIDTH * i;
        unsigned char *bitmap_row = bitmap + IMG_WIDTH / 8 * i;
        for (int j = 0; j < IMG_WIDTH / 8; j++) {
          bitmap_row[j] = 0;
          for (int k = 0; k < 8; k++) {
            if (row[j * 8 + k] >= DELTA) {
              bitmap_row[j] |= 1 << k;
              diff_img[count] = row[j * 8 + k];
              count++;
            }
          }
        }
      }
      outputs[i].write((char*)bitmap, IMG_SIZE / 2);
      outputs[i].write((char*)diff_img, count);

      frame_start_pos[(TOTAL_FRAMES * i + frame_no) * 2] = frame_start_pos_buffer[i];
      frame_start_pos[(TOTAL_FRAMES * i + frame_no) * 2 + 1] = count + IMG_SIZE / 2;
      frame_start_pos_buffer[i] += count + IMG_SIZE / 2;
    }
  }

  outputs[TOTAL_CAMS].write((char*)frame_start_pos, TOTAL_FRAMES * TOTAL_CAMS * sizeof(int) * 2);

  CHECK(cudaFree(d_img_buffer));
  CHECK(cudaFree(d_diff_img_buffer));
  CHECK(cudaFree(d_P));
  CHECK(cudaFree(d_XYZ));
  CHECK(cudaFree(d_UV));

  free(diff_img_buffer);
  free(img_buffer);
  free(bitmap);
  free(frame_start_pos);
}

void cuda_recover(ImageBuffer& image_buffer, int main_stream_no, int center_no, int stream_no, const int PIXEL_PER_THREAD, float *mat_P) {
  char frame_rate_str[64];
  char front_color[3] = { 0, 0, 0 };
  char back_color[3] = { 255, 255, 255 };
  CImgDisplay disp;

  const int IMG_RGB_SIZE = IMG_SIZE * IMG_CHANNEL;
  const int IMG_RGBD_SIZE = IMG_SIZE * (IMG_CHANNEL + 1);
  const int IMG_BUFFER_SIZE = IMG_RGBD_SIZE * 3; // main stream, center stream, reference stream
  unsigned char *diff_img_buffer;
  unsigned char *img_buffer;

  unsigned char *d_img_buffer, *d_img;
  float *d_P, *d_XYZ;
  int *d_UV;
  int image_no;
  CHECK(cudaMalloc(&d_img_buffer, IMG_BUFFER_SIZE));
  CHECK(cudaMalloc(&d_P, MAT_P_SIZE * TOTAL_CAMS * sizeof(float)));
  CHECK(cudaMalloc(&d_XYZ, IMG_SIZE * 3 * sizeof(float)));
  CHECK(cudaMalloc(&d_UV, IMG_SIZE * 2 * sizeof(int)));
  CHECK(cudaMemcpy(d_P, mat_P, MAT_P_SIZE * TOTAL_CAMS * sizeof(float), cudaMemcpyHostToDevice));

  // ----- group based recover ----- //
  for (int frame_no = 0; ; frame_no = (frame_no + 1) % TOTAL_FRAMES) {
    img_buffer = image_buffer.get_image();
    CHECK(cudaMemcpy(d_img_buffer, img_buffer, IMG_BUFFER_SIZE, cudaMemcpyHostToDevice));

    // ----- center stream recover ----- //
    if (center_no != main_stream_no) {
      coord_UVD_to_XYZ<<<IMG_HEIGHT, IMG_WIDTH>>>(d_img_buffer + IMG_RGB_SIZE, d_XYZ, GET_MAT_P(d_P, main_stream_no));
      CHECK(cudaDeviceSynchronize());
      coord_XYZ_to_UV <<<IMG_HEIGHT, IMG_WIDTH>>>(d_XYZ, d_UV, GET_MAT_P(d_P, center_no));
      CHECK(cudaDeviceSynchronize());
      recover_diff_image<<<IMG_HEIGHT, IMG_WIDTH / PIXEL_PER_THREAD >>>(d_img_buffer, d_img_buffer + IMG_RGBD_SIZE, d_UV);
      CHECK(cudaDeviceSynchronize());
    }
    if (stream_no != center_no) {
      coord_UVD_to_XYZ<<<IMG_HEIGHT, IMG_WIDTH>>>(d_img_buffer + IMG_RGBD_SIZE + IMG_RGB_SIZE, d_XYZ, GET_MAT_P(d_P, center_no));
      CHECK(cudaDeviceSynchronize());
      coord_XYZ_to_UV <<<IMG_HEIGHT, IMG_WIDTH>>>(d_XYZ, d_UV, GET_MAT_P(d_P, stream_no));
      CHECK(cudaDeviceSynchronize());
      recover_diff_image<<<IMG_HEIGHT, IMG_WIDTH / PIXEL_PER_THREAD >>>(d_img_buffer + IMG_RGBD_SIZE, d_img_buffer + IMG_RGBD_SIZE * 2, d_UV);
      CHECK(cudaDeviceSynchronize());
    }

    diff_img_buffer = img_buffer + IMG_RGBD_SIZE * 2;

    CHECK(cudaMemcpy(diff_img_buffer, d_img_buffer + IMG_RGBD_SIZE /** 2*/, IMG_RGBD_SIZE, cudaMemcpyDeviceToHost));

    // ----- display ----- //
    CImg<unsigned char> img(diff_img_buffer, IMG_WIDTH, IMG_HEIGHT, 1, IMG_CHANNEL, true);
    sprintf(frame_rate_str, "%.3f", disp.frames_per_second());
    img.draw_text(10, 10, frame_rate_str, front_color, back_color);
    //img.display();
    disp.display(img);
    if (disp.is_closed()) break;

    image_buffer.unload_image();
  }

  CHECK(cudaFree(d_img_buffer));
  CHECK(cudaFree(d_P));
  CHECK(cudaFree(d_XYZ));
  CHECK(cudaFree(d_UV));
}