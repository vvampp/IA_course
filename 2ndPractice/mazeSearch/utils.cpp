#include "utils.hpp"

int get_random_number(int max) { return rand() % max; }

int manhattan_distance(int row1, int col1, int row2, int col2){
  return abs(row1 - row2) + abs(col2 - col1);
}
