#ifndef S_ALGORITHMS_HPP
#define S_ALGORITHMS_HPP

#include <queue>
#include "grid.hpp"
#include "CONSTANTS.HPP"

struct CompareAStartCell {
  bool operator()(const std::pair<int,std::pair<int,int>>& a, const std::pair<int, std::pair<int,int>>& b){
    return a.first > b.first; // min heap implementation
  }
};


bool dfs_animation_step(Grid& grid, std::stack<std::pair<int,int>>& stack, bool& found_exit);
void dfs_animation(Grid& grid);

bool bfs_animation_step(Grid& grid, std::queue<std::pair<int,int>>& queue, bool& found_exit);
void bfs_animation(Grid& grid);

#endif
