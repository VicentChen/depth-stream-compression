#ifndef LOGGER_H__
#define LOGGER_H__

#include <iostream>
#include <string>
#include <fstream>

using namespace std;

class Logger {
public:
  static string get_time();
  static void prompt_debug(string message);
  static void prompt_info(string message);
  static void prompt_error(string message);

  static void debug_displayer_display(float ratio, float ratio_step, int frame_rate, int frame_no, bool play_flag);
};

#endif