#ifndef DSC_H__
#define DSC_H__

#include <CImg.h>
#include <fstream>

// camera calibration parameters
#define TOTAL_CAMS 8
#define TOTAL_FRAMES 100
#define MAT_A_ROW 3
#define MAT_A_COL 3
#define MAT_A_SIZE (MAT_A_ROW * MAT_A_COL)
#define MAT_Rt_ROW 3
#define MAT_Rt_COL 4
#define MAT_Rt_SIZE (MAT_Rt_ROW * MAT_Rt_COL)
#define MAT_P_ROW 4
#define MAT_P_COL 4
#define MAT_P_SIZE (MAT_P_ROW * MAT_P_COL)

#define  GET_MAT_A(Mat, N) (Mat + MAT_A_SIZE  * N)
#define  GET_MAT_P(Mat, N) (Mat + MAT_P_SIZE  * N)
#define GET_MAT_Rt(Mat, N) (Mat + MAT_Rt_SIZE * N)
#define  MAT_A(Mat, N, i, j) (GET_MAT_A (Mat, N)[i * MAT_A_COL  + j])
#define  MAT_P(Mat, N, i, j) (GET_MAT_P (Mat, N)[i * MAT_P_COL  + j])
#define MAT_Rt(Mat, N, i, j) (GET_MAT_Rt(Mat, N)[i * MAT_Rt_COL + j])
#define  MAT_Pn(P , i, j) (P [i * MAT_P_COL  + j])
#define  MAT_An(A , i, j) (A [i * MAT_A_COL  + j])
#define MAT_Rtn(Rt, i, j) (Rt[i * MAT_Rt_COL + j])

// image parameters
#define IMG_WIDTH 1024
#define IMG_HEIGHT 768
#define IMG_CHANNEL  3
#define IMG_SIZE (IMG_WIDTH * IMG_HEIGHT)
#define IMG_BYTE (IMG_SIZE * IMG_CHANNEL)
#define MinZ 44.0f
#define MaxZ 120.0f

using namespace cimg_library;

class Dsc {
private:
  float *mat_P;
  float *mat_A;
  float *mat_Rt;
  float *cam_direction;
  float angles[TOTAL_CAMS][TOTAL_CAMS];

  void compute_LSAS(int *group, int group_no, float *LSAS);
  void compute_GlSAS(float *GlSAS);
  float compute_angle(int no_1, int no_2);
  void get_init_group(int *group, bool *is_center, int group_count);
  void kmeans(int *group, int *new_group, int group_count, bool *is_center, bool *new_is_center, int kmeans_count);

public:
  Dsc();
  ~Dsc();

  void compute_calib_params(const char* filepath);
  void compress(const char* folderpath, int group_count, int main_stream_no, const int PIXEL_PER_THREAD, const unsigned char DELTA);
  void recover(const char* folderpath, int stream_no, const int PIXEL_PER_THREAD);

  void funct();
};

class ImageBuffer {
private:
  int buffer_size;
  unsigned char* buffer;
  bool stop_flag;
  int head;
  int tail;
  char *folderpath;
  int *frame_start_pos;
  unsigned char* expand_buf;

  std::ifstream files[TOTAL_CAMS];

public:
  int main_stream_no;
  int center_stream_no;
  int reference_stream_no;
  int frame_no;

  ImageBuffer(int buffer_size, const char *folderpath);
  ~ImageBuffer();

  inline bool is_full() { return (head + 1) % buffer_size == tail; }
  inline bool is_empty() { return head == tail; }

  void auto_load();
  void stop() { stop_flag = true; }
  void expand_image(int no, unsigned char *img_buf);
  void load_images();
  unsigned char* get_image();
  unsigned char* get_buffer();
  int get_image_no();
  void unload_image();
  void set_status(int main_stream_no, int center_stream_no, int reference_stream_no, int frame_no);
};

#endif
