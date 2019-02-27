#include "displayer.h"
#include "logger.h"
#include "dsc.h"
#include "dsc.cuh"

void preprocess() {

  const char *INPUT_PATTERN = "img/%s/cam%d/%s-cam%d-f%03d.%s";
  const char *OUTPUT_PATTERN = "img/%s/cam%d.dsc";
  // 1st %s
  const char *BREAK_DANCERS_FOLDER = "MSR3DVideo-Breakdancers";
  const char *BALLET_FOLDER = "MSR3DVideo-Ballet";
  // 2nd %s
  const char *COLOR = "color";
  const char *DEPTH = "depth";
  // 3rd %s
  const char *COLOR_SUFFIX = "jpg";
  const char *DEPTH_SUFFIX = "png";

  char filepath[256];

  CImg<unsigned char> img;
  for (int cam_no = 0; cam_no < TOTAL_CAMS; cam_no++) {
    sprintf(filepath, OUTPUT_PATTERN, BREAK_DANCERS_FOLDER, cam_no);
    ofstream output_file(filepath, ios::binary);

    for (int frame_no = 0; frame_no < TOTAL_FRAMES; frame_no++) {
      sprintf(filepath, INPUT_PATTERN, BREAK_DANCERS_FOLDER, cam_no, COLOR, cam_no, frame_no, COLOR_SUFFIX);
      img.load(filepath);
      output_file.write((const char*)img.data(), img.size());

      sprintf(filepath, INPUT_PATTERN, BREAK_DANCERS_FOLDER, cam_no, DEPTH, cam_no, frame_no, DEPTH_SUFFIX);
      img.load(filepath);
      output_file.write((const char*)img.data(), img.size());
    }
  }

  for (int cam_no = 0; cam_no < TOTAL_CAMS; cam_no++) {
    sprintf(filepath, OUTPUT_PATTERN, BALLET_FOLDER, cam_no);
    ofstream output_file(filepath, ios::binary);

    for (int frame_no = 0; frame_no < TOTAL_FRAMES; frame_no++) {
      sprintf(filepath, INPUT_PATTERN, BALLET_FOLDER, cam_no, COLOR, cam_no, frame_no, COLOR_SUFFIX);
      img.load(filepath);
      output_file.write((const char*)img.data(), img.size());

      sprintf(filepath, INPUT_PATTERN, BALLET_FOLDER, cam_no, DEPTH, cam_no, frame_no, DEPTH_SUFFIX);
      img.load(filepath);
      output_file.write((const char*)img.data(), img.size());
    }
  }
}

int main(int argc, char* argv[]) {
  //preprocess();

  //Displayer displayer;
  //displayer.display();

  Dsc dsc;
  dsc.compute_calib_params("img/MSR3DVideo-Breakdancers/calibParams-breakdancers.txt");
  CHECK(cudaSetDevice(0));
  CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  //dsc.compress("img/MSR3DVideo-Breakdancers", 3, 4, 8, 0);
  dsc.recover("img/MSR3DVideo-Breakdancers", 0, 8);
  CHECK(cudaDeviceReset());

  //unsigned char* img = (unsigned char*)malloc(IMG_SIZE * 4);
  //memset(img, 0, IMG_SIZE * 4);
  //ImageBuffer image_buffer(100, "img/MSR3DVideo-Breakdancers");
  //image_buffer.set_status(4, 1, 0, 0);
  //image_buffer.auto_load();
  //img = image_buffer.get_image();
  //for(int i = 0; i < 3; i++) {
  //  CImg<unsigned char> I(img + IMG_SIZE * 4 * i, 1024, 768, 1, 3, true);
  //  I.display();
  //}

  //free(img);
  //dsc.funct();

  //CImg<unsigned char> I("img/MSR3DVideo-Breakdancers/cam0/depth-cam0-f000.png");
  //I.display();

  return 0;
}
