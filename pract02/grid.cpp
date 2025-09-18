#include "grid.hpp"

#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/System/Vector2.hpp>
#include <iostream>
#include <random>
#include <tuple>
#include <utility>

#include "CONSTANTS.HPP"

Grid::Grid() {
  srand(time(0));
  highlighted_cell = std::make_pair(-1, -1);
}

void Grid::set_wall(int row1, int col1, int row2, int col2) {
  if (row1 == row2 and col1 == col2 + 1) {  // c1 is to the right of c2
    cells[col1][row1].add_left_wall();
    cells[col2][row2].add_right_wall();
  } else if (row1 == row2 and col1 == col2 - 1) {  // c1 is to the left of c2
    cells[col1][row1].add_right_wall();
    cells[col2][row2].add_left_wall();
  } else if (row1 == row2 + 1 and col1 == col2) {  // c1 is below c2
    cells[col1][row1].add_top_wall();
    cells[col2][row2].add_bottom_wall();
  } else if (row1 == row2 - 1 and col1 == col2) {  // c1 is above c2
    cells[col1][row1].add_bottom_wall();
    cells[col2][row2].add_top_wall();
  }
}

void Grid::remove_wall(int row1, int col1, int row2, int col2) {
  if (row1 == row2) {        // same row
    if (col1 == col2 + 1) {  // Cell 1 is to the right of Cell 2
      cells[row1][col1].remove_left_wall();
      cells[row2][col2].remove_right_wall();
    } else if (col1 == col2 - 1) {  // Cell 1 is to the left of Cell 2
      cells[row1][col1].remove_right_wall();
      cells[row2][col2].remove_left_wall();
    }
  } else if (col1 == col2) {  // Same column
    if (row1 == row2 + 1) {   // Cell 1 is below Cell2
      cells[row1][col1].remove_top_wall();
      cells[row2][col2].remove_bottom_wall();
    } else if (row1 == row2 - 1) {  // Cell 1 is above Cell2
      cells[row1][col1].remove_bottom_wall();
      cells[row2][col2].remove_top_wall();
    }
  }
}

void Grid::set_entry(int x, int y) {
  cells[x][y].set_as_entry();
  cells[x][y].remove_left_wall();
}

void Grid::set_exit(int x, int y) {
  cells[x][y].set_as_exit();
  cells[x][y].remove_right_wall();
}

Cell& Grid::get_cell(int x, int y) { return cells[x][y]; }

const Cell& Grid::get_cell(int x, int y) const { return cells[x][y]; }

void Grid::set_cell_as_visited(int x, int y) {
  get_cell(x, y).set_visited();
}

void Grid::set_highlighted_cell(int x, int y) {
  highlighted_cell = std::make_pair(x, y);
}

void Grid::reset_visits() {
  for (int i = 0; i < GRID_HEIGHT; ++i) {
    for (int j = 0; j < GRID_WIDTH; ++j) {
      get_cell(i, j).reset();
    }
  }
}

std::vector<std::pair<int, int>> Grid::get_neighbors(int row, int col, bool visited) const {
  std::vector<std::pair<int, int>> res;
  if (visited) {
    if (row > 0) {
      res.push_back(std::make_pair(row - 1, col));
    }
    if (row < GRID_HEIGHT - 1) {
      res.push_back(std::make_pair(row + 1, col));
    }
    if (col > 0) {
      res.push_back(std::make_pair(row, col - 1));
    }
    if (col < GRID_WIDTH - 1) {
      res.push_back(std::make_pair(row, col + 1));
    }
  } else {
    if (row > 0 and !get_cell(row - 1, col).check_if_visited()) {
      res.push_back(std::make_pair(row - 1, col));
    }
    if (row < GRID_HEIGHT - 1 and !get_cell(row + 1, col).check_if_visited()) {
      res.push_back(std::make_pair(row + 1, col));
    }
    if (col > 0 and !get_cell(row, col - 1).check_if_visited()) {
      res.push_back(std::make_pair(row, col - 1));
    }
    if (col < GRID_WIDTH - 1 and !get_cell(row, col + 1).check_if_visited()) {
      res.push_back(std::make_pair(row, col + 1));
    }
  }
  return res;
}

std::vector<std::pair<int, int>> Grid::get_neighbors_search(int row, int col) const {
  std::vector<std::pair<int, int>> result;

  // Check top neighbor (row - 1, col)
  if (row > 0 && !cells[row][col].has_top_wall()) {
    result.push_back(std::make_pair(row - 1, col));
  }
  // Check bottom neighbor (row + 1, col)
  if (row < GRID_HEIGHT - 1 && !cells[row][col].has_bottom_wall()) {
    result.push_back(std::make_pair(row + 1, col));
  }
  // Check left neighbor (row, col - 1)
  if (col > 0 && !cells[row][col].has_left_wall()) {
    result.push_back(std::make_pair(row, col - 1));
  }
  // Check right neighbor (row, col + 1)
  if (col < GRID_WIDTH - 1 && !cells[row][col].has_right_wall()) {
    result.push_back(std::make_pair(row, col + 1));
  }
  return result;
}

bool Grid::get_wall_between_cells(int x1, int y1, int x2, int y2) {
  if (abs(x1 - x2) + abs(y1 - y2) != 1) {
    return true;
  }

  if (x1 == x2) {
    if (y1 > y2) {
      return cells[x1][y1].has_right_wall();
    } else {
      return cells[x1][y1].has_left_wall();
    }
  } else {
    if (x1 > x2) {
      return cells[x1][y1].has_bottom_wall();
    } else {
      return cells[x1][y1].has_top_wall();
    }
  }
}

void Grid::prim_maze() {
  std::random_device rd;
  std::mt19937 g(rd());

  // Start with a random cell
  int start_x = get_random_number(GRID_HEIGHT);
  int start_y = get_random_number(GRID_WIDTH);
  cells[start_x][start_y].set_visited();

  // Add the walls of the cell to the list of walls
  std::vector<std::tuple<int, int, int, int>> walls;
  if (start_x > 0) walls.push_back(std::make_tuple(start_x, start_y, start_x - 1, start_y));
  if (start_x < GRID_HEIGHT - 1) walls.push_back(std::make_tuple(start_x, start_y, start_x + 1, start_y));
  if (start_y > 0) walls.push_back(std::make_tuple(start_x, start_y, start_x, start_y - 1));
  if (start_y < GRID_WIDTH - 1) walls.push_back(std::make_tuple(start_x, start_y, start_x, start_y + 1));

  // Loop until there are no walls left in the list
  while (!walls.empty()) {
    // : Randomly select a wall from the list
    std::shuffle(walls.begin(), walls.end(), g);
    auto wall = walls.back();
    walls.pop_back();

    int x1 = std::get<0>(wall);
    int y1 = std::get<1>(wall);
    int x2 = std::get<2>(wall);
    int y2 = std::get<3>(wall);

    // If only one of the two cells divided by the wall is visited
    if (cells[x2][y2].check_if_visited() != cells[x1][y1].check_if_visited()) {
      // Remove the wall between the two cells
      remove_wall(x1, y1, x2, y2);

      // Mark the unvisited cell as part of the maze
      cells[x2][y2].set_visited();

      // Add the neighboring walls of the cell to the list
      if (x2 > 0 && !cells[x2 - 1][y2].check_if_visited()) walls.push_back(std::make_tuple(x2, y2, x2 - 1, y2));
      if (x2 < GRID_HEIGHT - 1 && !cells[x2 + 1][y2].check_if_visited()) walls.push_back(std::make_tuple(x2, y2, x2 + 1, y2));
      if (y2 > 0 && !cells[x2][y2 - 1].check_if_visited()) walls.push_back(std::make_tuple(x2, y2, x2, y2 - 1));
      if (y2 < GRID_WIDTH - 1 && !cells[x2][y2 + 1].check_if_visited()) walls.push_back(std::make_tuple(x2, y2, x2, y2 + 1));
    }
  }
}

bool Grid::prim_maze_animation_step(std::vector<std::tuple<int, int, int, int>>& walls) {
  if (walls.empty()) return false;

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(walls.begin(), walls.end(), g);

  auto wall = walls.back();
  walls.pop_back();

  int x1 = std::get<0>(wall);
  int y1 = std::get<1>(wall);
  int x2 = std::get<2>(wall);
  int y2 = std::get<3>(wall);

  if (cells[x2][y2].check_if_visited() != cells[x1][y1].check_if_visited()) {
    remove_wall(x1, y1, x2, y2);
    cells[x2][y2].set_visited();

    if (x2 > 0 && !cells[x2 - 1][y2].check_if_visited()) walls.push_back(std::make_tuple(x2, y2, x2 - 1, y2));
    if (x2 < GRID_HEIGHT - 1 && !cells[x2 + 1][y2].check_if_visited()) walls.push_back(std::make_tuple(x2, y2, x2 + 1, y2));
    if (y2 > 0 && !cells[x2][y2 - 1].check_if_visited()) walls.push_back(std::make_tuple(x2, y2, x2, y2 - 1));
    if (y2 < GRID_WIDTH - 1 && !cells[x2][y2 + 1].check_if_visited()) walls.push_back(std::make_tuple(x2, y2, x2, y2 + 1));

    return true;
  }

  return true;
}

void Grid::draw(sf::RenderWindow& window) {
  float pos_x = WIDTH / 2 - GRID_WIDTH * SQUARE_SIZE / 2;
  float pos_y = HEIGHT / 2 - GRID_HEIGHT * SQUARE_SIZE / 2 - 50;

  for (int i = 0; i < GRID_HEIGHT; ++i) {
    for (int j = 0; j < GRID_WIDTH; ++j) {
      if (i == highlighted_cell.first && j == highlighted_cell.second)
        cells[i][j].draw(
            window,
            sf::Vector2f(pos_x + SQUARE_SIZE * j, pos_y + SQUARE_SIZE * i),
            true);
      else
        cells[i][j].draw(
            window,
            sf::Vector2f(pos_x + SQUARE_SIZE * j, pos_y + SQUARE_SIZE * i));
    }
  }
}

void Grid::draw_path(sf::RenderWindow& window) {
  sf::Color path_color = sf::Color::Green;
  sf::RectangleShape path_rect(sf::Vector2f(SQUARE_SIZE, SQUARE_SIZE));
  path_rect.setFillColor(path_color);

  float pos_x = WIDTH / 2 - GRID_WIDTH * SQUARE_SIZE / 2;
  float pos_y = HEIGHT / 2 - GRID_HEIGHT * SQUARE_SIZE / 2 - 50;

  std::pair<int, int> current = {GRID_HEIGHT - 1, GRID_WIDTH - 1};
  while (current.first != 0 || current.second != 0) {
      // Draw the current cell as part of the path
      path_rect.setPosition(pos_x + SQUARE_SIZE * current.second, pos_y + SQUARE_SIZE * current.first);
      window.draw(path_rect);

      // Get the coordinates of the parent cell
      std::pair<int, int> parent_dir = cells[current.first][current.second].get_direction();
      
      // Move to the parent cell for the next iteration
      current = parent_dir;
    }
    // Draw the starting cell (0, 0)
    path_rect.setPosition(pos_x, pos_y);
    window.draw(path_rect);
}

