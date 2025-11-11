#ifndef S_ALGORITHMS_HPP
#define S_ALGORITHMS_HPP

#include "CONSTANTS.HPP"
#include "grid.hpp"
#include <queue>
#include <vector>

struct CompareAStarCell {
    bool operator()(const std::pair<int, std::pair<int, int>> &a,
                    const std::pair<int, std::pair<int, int>> &b) {
        return a.first > b.first; // min heap implementation
    }
};

bool dfs_animation_step(Grid &grid, std::stack<std::pair<int, int>> &stack, bool &found_exit);
void dfs_animation(Grid &grid);

bool bfs_animation_step(Grid &grid, std::queue<std::pair<int, int>> &queue, bool &found_exit);
void bfs_animation(Grid &grid);

bool a_star_animation_step(
    Grid &grid,
    std::priority_queue<std::pair<int, std::pair<int, int>>,
                        std::vector<std::pair<int, std::pair<int, int>>>, CompareAStarCell> &pq,
    std::vector<std::vector<int>> &g_scores, bool &found_exit);
void a_star_animation(Grid &grid);
#endif
