#pragma once
#ifndef DISPLAYER_H__
#define DISPLAYER_H__

#include <CImg.h>

using namespace cimg_library;

class Displayer {
private:
  static const int DEFAULT_WIDTH = 1024;
  static const int DEFAULT_HEIGHT = 768;

  CImgDisplay canvas;

public:
  Displayer();
  Displayer(int width, int height);
  void display();
};

#endif