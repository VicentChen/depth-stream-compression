#include "displayer.h"
#include "logger.h"

Displayer::Displayer() {
  canvas.resize(DEFAULT_WIDTH, DEFAULT_HEIGHT);
}

Displayer::Displayer(int width, int height) {
  canvas.resize(width, height);
}

void Displayer::display() {
  canvas.show();

  float ratio = 1; // initial left ratio of interpolation
  float ratio_step = 0.01; // initial ratio step
  int frame_rate = 60; // initial frame rate
  int frame_no = 0; // initial frame no
  bool play_flag = true;

  while (!canvas.is_closed()) {
    // TODO: get key
    unsigned int key = canvas.key();
    if (key == cimg::keyESC) break;

    switch (key) {
      // view direction control
      case cimg::keyA: ratio += ratio_step; break;
      case cimg::keyD: ratio -= ratio_step; break;
      // frame rate control
      case cimg::keyPAGEUP: frame_rate++; break;
      case cimg::keyPAGEDOWN: frame_rate--; break;
      // video control
      case cimg::keyW: if(!play_flag) frame_no++; break;
      case cimg::keyS: if(!play_flag) frame_no--; break;
      case cimg::keySPACE: play_flag = !play_flag; break;

      case cimg::keyENTER:
        Logger::debug_displayer_display(ratio, ratio_step, frame_rate, frame_no, play_flag);
        break;

      default: break;
    }

    // TODO: get image, refresh ratio

    // TODO: display

    canvas.wait(1000 / frame_rate);
  }

  cout << "???" << endl;
}
