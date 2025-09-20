#ifndef UTILS_HPP
#define UTILS_HPP

#include "CONSTANTS.HPP"
#include "cell.hpp"
#include <time.h>
#include <utility>
#include <vector>
typedef struct node node;

int get_random_number(int max);

int manhattan_distance(int row1, int col1, int row2, int col2);

struct node {
  int prev_x, prev_y, current_x, current_y;
  node &operator=(const node &a) {
    prev_x = a.prev_x;
    prev_y = a.prev_y;
    current_x = a.current_x;
    current_y = a.current_y;
    return *this;
  }
};

#endif
