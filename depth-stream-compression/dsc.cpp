#include "dsc.h"
#include "logger.h"
#include "dsc.cuh"

#include <thread>
#include <Windows.h>

void Dsc::compute_LSAS(int* group, int group_no, float* LSAS) {
  int LSAS_idx = 0;
  for (int i = 0; i < TOTAL_CAMS; i++) {
    if (group[i] != group_no) continue;
    LSAS[LSAS_idx] = 0;
    for (int k = 0; k < TOTAL_CAMS; k++) {
      if (group[k] != group_no) continue;
      LSAS[LSAS_idx] += angles[i][k] * angles[i][k];
    }
    LSAS_idx++;
  }
}

void Dsc::compute_GlSAS(float* GlSAS) {
  for (int i = 0; i < TOTAL_CAMS; i++) {
    GlSAS[i] = 0;
    for (int k = 0; k < TOTAL_CAMS; k++) {
      GlSAS[i] += angles[i][k] * angles[i][k];
    }
  }
}

float Dsc::compute_angle(int no_1, int no_2) {
  float x1 = cam_direction[no_1 * 3 + 0];
  float y1 = cam_direction[no_1 * 3 + 1];
  float z1 = cam_direction[no_1 * 3 + 2];
  float x2 = cam_direction[no_2 * 3 + 0];
  float y2 = cam_direction[no_2 * 3 + 1];
  float z2 = cam_direction[no_2 * 3 + 2];

  float cos_a = (x1 * x2 + y1 * y2 + z1 * z2)
    / sqrt(x1 * x1 + y1 * y1 + z1 * z1)
    / sqrt(x2 * x2 + y2 * y2 + z2 * z2);

  return acosf(cos_a) / cimg::PI * 180;
}

void Dsc::get_init_group(int* group, bool* is_center, int group_count) {
  float GlSAS[TOTAL_CAMS];
  compute_GlSAS(GlSAS);

  // streams are already sorted
  // for simplicity select the first candidate as initial center stream

  // get size range of a group
  int group_size_min = TOTAL_CAMS / group_count;
  int group_size_max = TOTAL_CAMS / group_count + 1;
  if (group_size_min * group_count == TOTAL_CAMS) group_size_min--;

  int curr_sum = 0;
  int group_idx = 0;
  for (int i = 0; i < group_count; i++) {
    int curr_group_size = group_size_min;
    while ((group_count - i - 1) * group_size_max < (TOTAL_CAMS - curr_sum - curr_group_size)) {
      curr_group_size++;
    }

    // set init group
    for (int k = 0; k < curr_group_size; k++) {
      is_center[group_idx] = false;
      group[group_idx++] = i;
    }

    // set init center stream
    is_center[curr_sum + curr_group_size / 2] = true;

    curr_sum += curr_group_size;
  }
}

void Dsc::kmeans(int* group, int* new_group, int group_count, bool* is_center, bool* new_is_center, int kmeans_count) {
  if (kmeans_count >= 100) {
    for (int i = 0; i < TOTAL_CAMS; i++) {
      new_group[i] = group[i];
      new_is_center[i] = is_center[i];
    }
    return;
  }
  kmeans_count++;

  // ----- generate new center stream ----- //
  int new_center[TOTAL_CAMS];
  int new_center_idx = 0;

  // reset center streams
  for (int i = 0; i < TOTAL_CAMS; i++) new_is_center[i] = false;

  // get group size
  int group_size[TOTAL_CAMS];
  for (int i = 0; i < group_count; i++) group_size[i] = 0;
  for (int i = 0; i < TOTAL_CAMS; i++) group_size[group[i]]++;

  // compute LSAS
  float LSAS[TOTAL_CAMS];
  for (int group_no = 0; group_no < group_count; group_no++) {
    compute_LSAS(group, group_no, LSAS);
    int min_LSAS_idx = 0;
    for (int i = 0; i < group_size[group_no]; i++) {
      if (LSAS[i] < LSAS[min_LSAS_idx]) min_LSAS_idx = i;
    }
    // get new center stream
    int min_stream_no = 0;
    int group_stream_index = 0;
    for (int i = 0; i < TOTAL_CAMS; i++) {
      if (group[i] == group_no) {
        if (group_stream_index == min_LSAS_idx) {
          min_stream_no = i;
          break;
        }
        group_stream_index++;
      }
    }
    new_is_center[min_stream_no] = true;
    new_center[new_center_idx++] = min_stream_no;
  }

  // ----- generate new group ----- //
  for (int stream_no = 0; stream_no < TOTAL_CAMS; stream_no++) {
    float min_angle = FLT_MAX;
    for (int group_no = 0; group_no < group_count; group_no++) {
      float angle = abs(angles[stream_no][new_center[group_no]]);
      if (angle < min_angle) {
        min_angle = angle;
        new_group[stream_no] = group_no;
      }
    }
  }

  // ----- check if stable ----- //
  bool stable = true;
  for (int i = 0; i < TOTAL_CAMS; i++) {
    if (new_is_center[i] != is_center[i] || new_group[i] != group[i]) {
      stable = false;
      break;
    }
  }

  if (!stable) {
    kmeans(new_group, group, group_count, new_is_center, is_center, kmeans_count);
  }
  else {
    for (int i = 0; i < TOTAL_CAMS; i++) {
      new_group[i] = group[i];
      new_is_center[i] = is_center[i];
    }
  }
}

Dsc::Dsc() {
  mat_P         = (float*)malloc(sizeof(float) * MAT_P_SIZE  * TOTAL_CAMS);
  mat_A         = (float*)malloc(sizeof(float) * MAT_A_SIZE  * TOTAL_CAMS);
  mat_Rt        = (float*)malloc(sizeof(float) * MAT_Rt_SIZE * TOTAL_CAMS);
  cam_direction = (float*)malloc(sizeof(float) * TOTAL_CAMS * 3);
}

Dsc::~Dsc() {
  free(mat_P        );
  free(mat_A        );
  free(mat_Rt       );
  free(cam_direction);
}

void Dsc::compute_calib_params(const char* filepath) {
  // ----- open file ----- //
  ifstream file(filepath);
  if (!file) {
    Logger::prompt_error("Cannot open camera calibration file");
    exit(-1);
  }

  // alias
  float *P = mat_P, *A = mat_A, *Rt = mat_Rt;

  // ----- read from file ----- //
  float dummy;
  int A_i = 0, Rt_i = 0;
  for (int i = 0; i < TOTAL_CAMS; i++) {
    file >> dummy; // skip Camera Number
    for (int k = 0; k < MAT_A_SIZE; k++) file >> A[A_i++]; // read Intrinsic Matrix
    file >> dummy; // skip barrel distortion
    file >> dummy; // skip barrel distortion
    for (int k = 0; k < MAT_Rt_SIZE; k++) file >> Rt[Rt_i++]; // read Rotation Matrix
  }

  // ----- compute projection matrix ----- //
  float *v;
  for (int N = 0; N < TOTAL_CAMS; N++) {
    for (int i = 0; i < MAT_Rt_ROW; i++) {
      for (int j = 0; j < MAT_Rt_COL; j++) {
        v = &MAT_P(P, N, i, j);
        *v = 0;
        for (int k = 0; k < MAT_A_ROW; k++) {
          *v += MAT_A(A, N, i, k) * MAT_Rt(Rt, N, k, j);
        }
      }
    }

    MAT_P(P, N, 3, 0) = 0; MAT_P(P, N, 3, 1) = 0;
    MAT_P(P, N, 3, 2) = 0; MAT_P(P, N, 3, 3) = 1;
  }

  // ----- compute camera direction ----- //
  for (int i = 0; i < TOTAL_CAMS; i++) {
    float sub_det[MAT_Rt_ROW][MAT_Rt_ROW];
    for (int r = 0; r < MAT_Rt_ROW; r++) {
      for (int c = 0; c < MAT_Rt_ROW; c++) {
        float a1 = MAT_Rt(Rt, i, (r + 1 + MAT_Rt_ROW) % MAT_Rt_ROW, (c + 1 + MAT_Rt_ROW) % MAT_Rt_ROW);
        float a2 = MAT_Rt(Rt, i, (r - 1 + MAT_Rt_ROW) % MAT_Rt_ROW, (c + 1 + MAT_Rt_ROW) % MAT_Rt_ROW);
        float b1 = MAT_Rt(Rt, i, (r + 1 + MAT_Rt_ROW) % MAT_Rt_ROW, (c - 1 + MAT_Rt_ROW) % MAT_Rt_ROW);
        float b2 = MAT_Rt(Rt, i, (r - 1 + MAT_Rt_ROW) % MAT_Rt_ROW, (c - 1 + MAT_Rt_ROW) % MAT_Rt_ROW);
        sub_det[c][r] = (a1 * b2 - a2 * b1);
      }
    }

    float det = 0;
    for (int c = 0; c < MAT_Rt_ROW; c++) {
      det += MAT_Rt(Rt, i, 0, c) * sub_det[c][0];
    }

    for (int r = 0; r < MAT_Rt_ROW; r++) {
      for (int c = 0; c < MAT_Rt_ROW; c++) {
        sub_det[r][c] /= det;
      }
    }

    cam_direction[0 + i * 3] = sub_det[0][2];
    cam_direction[1 + i * 3] = sub_det[1][2];
    cam_direction[2 + i * 3] = sub_det[2][2];
  }

  // ----- compute all angles ----- //
  for (int i = 0; i < TOTAL_CAMS; i++) {
    for (int k = 0; k < TOTAL_CAMS; k++) {
      if (k == i) {
        angles[i][k] = 0;
        continue;
      }
      if (k < i)
        angles[i][k] = compute_angle(i, k) * -1;
      else
        angles[i][k] = compute_angle(i, k);
    }
  }
}

void Dsc::compress(const char* folderpath, int group_count, int main_stream_no, const int PIXEL_PER_THREAD, const unsigned char DELTA) {
  // ----- get groups ----- //
  int group[TOTAL_CAMS], new_group[TOTAL_CAMS];
  bool is_center[TOTAL_CAMS], new_is_center[TOTAL_CAMS];
  get_init_group(group, is_center, group_count);
  kmeans(group, new_group, group_count, is_center, new_is_center, 100);

  char filepath[256];
  ifstream files[TOTAL_CAMS];
  ofstream outputs[TOTAL_CAMS + 1];
  for (int i = 0; i < TOTAL_CAMS; i++) {
    sprintf(filepath, "%s/cam%d.dsc", folderpath, i);
    files[i].open(filepath, ios::binary);
    sprintf(filepath, "%s/cam%d-diff-compressed.dsc", folderpath, i);
    outputs[i].open(filepath, ios::binary);
  }
  sprintf(filepath, "%s/cam-start-pos.dsc", folderpath);
  outputs[TOTAL_CAMS].open(filepath, ios::binary);

  cuda_compress(files, outputs, group, is_center, group_count, main_stream_no, PIXEL_PER_THREAD, mat_P, DELTA);

  // TODO: save groups
}


void Dsc::recover(const char* folderpath, int stream_no, const int PIXEL_PER_THREAD) {
  ifstream files[3];

  // TODO: read groups

  ImageBuffer image_buffer(100, folderpath);
  image_buffer.set_status(4, 1, 0, 0);
  image_buffer.auto_load();
  cuda_recover(image_buffer, 4, 1, stream_no, PIXEL_PER_THREAD, mat_P);
  image_buffer.stop();
}

void Dsc::funct() {
  ifstream file("img/MSR3DVideo-Breakdancers/cam0-compressed.dsc", ios::binary);

  const int DELTA = 25;

  int *start_pos = (int*)malloc(IMG_HEIGHT * 4 * sizeof(int));
  unsigned char *bitmap = (unsigned char*)malloc(IMG_SIZE / 2);
  unsigned char *buf = (unsigned char*)malloc(IMG_SIZE * 4);
  unsigned char *out_buf = (unsigned char*)malloc(IMG_SIZE * 4);
  memset(out_buf, 0, IMG_SIZE * 4);

  file.read((char*)buf, IMG_SIZE * 4);

  int count = 0;
  for (int i = 0; i < IMG_HEIGHT * 4; i++) {
    unsigned char *row = buf + IMG_WIDTH * i;
    unsigned char *bitmap_row = bitmap + IMG_WIDTH / 8 * i;
    start_pos[i] = count;
    for (int j = 0; j < IMG_WIDTH / 8; j++) {
      bitmap_row[j] = 0;
      for (int k = 0; k < 8; k++) {
        if (row[j * 8 + k] >= DELTA) {
          bitmap_row[j] |= 1 << k;
          buf[count] = row[j * 8 + k];
          count++;
        }
      }
    }
  }

  ifstream new_file("img/MSR3DVideo-Breakdancers/cam0-diff-compressed.dsc", ios::binary);
  new_file.read((char*)out_buf, IMG_SIZE / 2);
  int error = 0;
  for(int i = 0; i < IMG_SIZE / 2; i++) {
    if (out_buf[i] != bitmap[i]) {
      cout << (int)out_buf[i] << " " << (int)bitmap[i] << endl;
      error++;
    }
  }
  cout << error << endl;;

  //for (int i = 0; i < IMG_HEIGHT * 4; i++) {
  //  unsigned char *start = buf + start_pos[i];
  //  unsigned char *row = out_buf + IMG_WIDTH * i;
  //  for (int j = 0; j < IMG_WIDTH / 8; j++) {
  //    unsigned char *bitmap_row = bitmap + IMG_WIDTH / 8 * i;
  //    for (int k = 0; k < 8; k++) {
  //      if (bitmap_row[j] & (1 << k)) {
  //        row[j * 8 + k] = *start;
  //        start++;
  //      }
  //    }
  //  }
  //}

  //file.seekg(0);
  //file.read((char*)buf, IMG_SIZE * 4);

  //int error = 0;
  //for (int i = 0; i < IMG_SIZE * 4; i++) {
  //  if (buf[i] < DELTA) buf[i] = 0;
  //  if (buf[i] != out_buf[i]) error++;
  //}
  //cout << count << " " << count / 4.0 / IMG_SIZE << endl;
  //cout << error << endl;
}

ImageBuffer::ImageBuffer(int buffer_size, const char *folderpath) {
  this->buffer_size = buffer_size;

  CHECK(cudaHostAlloc(&buffer, (buffer_size + 1) * IMG_SIZE * 4 * 3, cudaHostAllocMapped));

  stop_flag = false;

  char filepath[256];
  for (int i = 0; i < TOTAL_CAMS; i++) {
    sprintf(filepath, "%s/cam%d-diff-compressed.dsc", folderpath, i);
    files[i].open(filepath, ios::binary);
  }

  head = 0;
  tail = 0;

  frame_start_pos = (int*)malloc(TOTAL_FRAMES * TOTAL_CAMS * sizeof(int) * 2);
  sprintf(filepath, "%s/cam-start-pos.dsc", folderpath);
  ifstream file(filepath, ios::binary);
  file.read((char*)frame_start_pos, TOTAL_FRAMES * TOTAL_CAMS * sizeof(int) * 2);

  expand_buf = (unsigned char*)malloc(IMG_SIZE * 4 + IMG_SIZE / 2);
}

ImageBuffer::~ImageBuffer() {
  free(frame_start_pos);
  free(expand_buf);
}

void ImageBuffer::auto_load() {
  thread t(&ImageBuffer::load_images, this);
  t.detach();
}

void ImageBuffer::expand_image(int no, unsigned char *img_buf) {

  int start_pos = frame_start_pos[(TOTAL_FRAMES * no + frame_no) * 2];
  int size = frame_start_pos[(TOTAL_FRAMES * no + frame_no) * 2 + 1];
  files[no].seekg(start_pos);
  files[no].read((char*)expand_buf, size);

  unsigned char* bitmap = expand_buf;
  unsigned char* compressed_img_buf = expand_buf + IMG_SIZE / 2;

  int expand_buf_i = 0;

  for(int i = 0; i < IMG_SIZE / 2; i++) {
    unsigned char *group = img_buf + i * 8;
    for (int k = 0; k < 8; k++) {
      if(bitmap[i] & (1 << k)) {
        group[k] = compressed_img_buf[expand_buf_i++];
      }
    }
  }
}

void ImageBuffer::load_images() {
  const int IMG_RGBD_SIZE = IMG_SIZE * 4;
  const int IMG_BUFFER_ELEMENT_SIZE = IMG_RGBD_SIZE * 3;

  while(!stop_flag) {

    if (!is_full()) {

      //files[main_stream_no].seekg(frame_no * IMG_RGBD_SIZE);
      //files[main_stream_no].read((char*)buffer + IMG_BUFFER_ELEMENT_SIZE * head, IMG_RGBD_SIZE);
      //files[center_stream_no].seekg(frame_no * IMG_RGBD_SIZE);
      //files[center_stream_no].read((char*)buffer + IMG_BUFFER_ELEMENT_SIZE * head + IMG_RGBD_SIZE, IMG_RGBD_SIZE);
      //files[reference_stream_no].seekg(frame_no * IMG_RGBD_SIZE);
      //files[reference_stream_no].read((char*)buffer + IMG_BUFFER_ELEMENT_SIZE * head + IMG_RGBD_SIZE * 2, IMG_RGBD_SIZE);
      memset(buffer + IMG_BUFFER_ELEMENT_SIZE * head, 0, IMG_RGBD_SIZE * 3);
      expand_image(main_stream_no, buffer + IMG_BUFFER_ELEMENT_SIZE * head);
      expand_image(center_stream_no, buffer + IMG_BUFFER_ELEMENT_SIZE * head + IMG_RGBD_SIZE);
      expand_image(reference_stream_no, buffer + IMG_BUFFER_ELEMENT_SIZE * head + IMG_RGBD_SIZE * 2);

      frame_no = (frame_no + 1) % TOTAL_FRAMES;
      head = (head + 1) % buffer_size;
    }
  }

  CHECK(cudaFreeHost(buffer));
}

unsigned char* ImageBuffer::get_image() {
  while (is_empty()) { Sleep(5); }

  return buffer + IMG_SIZE * 4 * 3 * tail;
}

unsigned char* ImageBuffer::get_buffer() {
  return buffer;
}

int ImageBuffer::get_image_no() {
  while (is_empty()) { Sleep(5); }
  return tail;
}

void ImageBuffer::set_status(int main_stream_no, int center_stream_no, int reference_stream_no, int frame_no) {
  this->main_stream_no = main_stream_no;
  this->center_stream_no = center_stream_no;
  this->reference_stream_no = reference_stream_no;
  this->frame_no = frame_no;
}

void ImageBuffer::unload_image() {
  if (!is_empty()) {
    tail = (tail + 1) % buffer_size;
  }
}
