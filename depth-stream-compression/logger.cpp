#include "logger.h"

string Logger::get_time() {
  return "[xx:xx:xx]";
}

void Logger::prompt_debug(string message) {
  string time_str = get_time();
  cout << "[DEBUG] " << time_str << ":" << message << endl;
}

void Logger::prompt_info(string message) {
  string time_str = get_time();
  cout << "[INFO ] " << time_str << ":" << message << endl;
}

void Logger::prompt_error(string message) {
  string time_str = get_time();
  cout << "[ERROR] " << time_str << ":" << message << endl;
}

void Logger::debug_displayer_display(float ratio, float ratio_step, int frame_rate, int frame_no, bool play_flag) {
  cout << "--Displayer-display()--" << endl;
  cout << "ratio: " << ratio << endl;
  cout << "frame rate: " << frame_rate << endl;
  cout << "play flag: " << play_flag << endl;
  cout << "frame_no: " << frame_no << endl;
  cout << "-----------------------" << endl;
}
