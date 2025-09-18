#ifndef GRID_HPP
#define GRID_HPP

#include <SFML/Graphics/RenderWindow.hpp>
#include <algorithm>
#include <random>
#include <stack>

#include "CONSTANTS.HPP"
#include "cell.hpp"
#include "utils.hpp"

class Grid {
 private:
  Cell cells[GRID_HEIGHT][GRID_WIDTH];
  std::pair<int, int> highlighted_cell;

 public:
  Grid();

  void set_wall(int row1, int col1, int row2, int col2);
  void remove_wall(int row1, int col1, int row2, int col2);
  void set_entry(int x, int y);
  void set_exit(int x, int y);

  Cell& get_cell(int x, int y);
  const Cell& get_cell(int x, int y) const;
  void set_cell_as_visited(int x, int y);
  void set_highlighted_cell(int x, int y);
  void reset_visits();

  std::vector<std::pair<int, int>> get_neighbors(int x, int y, bool visited = false) const;
  std::vector<std::pair<int,int>> get_neighbors_search(int x, int y) const;
  bool get_wall_between_cells(int x1, int y1, int x2, int y2);

  void prim_maze();
  bool prim_maze_animation_step(std::vector<std::tuple<int, int, int, int>> &walls);

  void draw(sf::RenderWindow &window);
  void draw_path(sf::RenderWindow& window);

  int manhattan_distance(int row1, int col1, int row2, int col2);
};
#endif // !GRID_HPP
