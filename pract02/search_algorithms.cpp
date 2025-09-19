#include "search_algorithms.hpp"

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/System/Sleep.hpp>
#include <SFML/System/Time.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Window/VideoMode.hpp>
#include <iostream>
#include <list>
#include <queue>

#include "CONSTANTS.HPP"
#include "utils.hpp"

bool dfs_animation_step(Grid& grid, std::stack<std::pair<int, int>>& stack, bool& found_exit) {
  if (stack.empty()) {
    return false;
  }

  std::pair<int, int> current_cell = stack.top();
  stack.pop();

  int row = current_cell.first;
  int col = current_cell.second;

  if (grid.get_cell(row, col).check_if_visited()) {
    return true;
  }

  grid.set_cell_as_visited(row, col);
  grid.set_highlighted_cell(row, col);

  if (grid.get_cell(row, col).is_exit()) {
    found_exit = true;
    return false;
  }

  auto neighbors = grid.get_neighbors_search(row, col);

  for (const auto& neighbor : neighbors) {
    int n_row = neighbor.first;
    int n_col = neighbor.second;
    if (!grid.get_cell(n_row, n_col).check_if_visited()) {
      grid.get_cell(n_row, n_col).set_direction(row, col);
      stack.push(neighbor);
    }
  }
  return true;
}

void dfs_animation(Grid& grid) {
  sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Depth First Search");
  std::stack<std::pair<int, int>> stack;
  bool found_exit = false;

  grid.set_highlighted_cell(0, 0);
  stack.push({0, 0});

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed)
        window.close();
    }
    window.clear();

    if (!found_exit) {
      dfs_animation_step(grid, stack, found_exit);
      sf::sleep(sf::microseconds(15000));
    } else {
      grid.set_highlighted_cell(-1, -1);
      grid.draw_path(window);
      sf::sleep(sf::microseconds(20));
    }

    grid.draw(window);
    window.display();
  }
}

bool bfs_animation_step(Grid& grid, std::queue<std::pair<int, int>>& queue, bool& found_exit) {
  if (queue.empty()) {
    return false;
  }
  std::pair<int, int> current_cell = queue.front();
  queue.pop();

  int row = current_cell.first;
  int col = current_cell.second;

  if (grid.get_cell(row, col).check_if_visited()) {
    return true;
  }

  grid.set_cell_as_visited(row, col);
  grid.set_highlighted_cell(row, col);

  if (grid.get_cell(row, col).is_exit()) {
    found_exit = true;
    return false;
  }

  auto neighbors = grid.get_neighbors_search(row, col);

  for (const auto& neighbor : neighbors) {
    int n_row = neighbor.first;
    int n_col = neighbor.second;
    if (!grid.get_cell(n_row, n_col).check_if_visited()) {
      grid.get_cell(n_row, n_col).set_direction(row, col);
      queue.push(neighbor);
    }
  }
  return true;
}

void bfs_animation(Grid& grid) {
  sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Breadth First Search");
  std::queue<std::pair<int, int>> queue;
  bool found_exit = false;

  grid.set_highlighted_cell(0, 0);
  queue.push({0, 0});

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed)
        window.close();
    }
    window.clear();

    if (!found_exit) {
      bfs_animation_step(grid, queue, found_exit);
      sf::sleep(sf::microseconds(15000));
    } else {
      grid.set_highlighted_cell(-1, -1);
      grid.draw_path(window);
      sf::sleep(sf::milliseconds(20));
    }
    grid.draw(window);
    window.display();
  }
}

bool a_star_animation_step(Grid& grid, std::priority_queue<std::pair<int, std::pair<int, int>>, std::vector<std::pair<int, std::pair<int, int>>>, CompareAStarCell>& pq, std::vector<std::vector<int>>& g_scores, bool& found_exit){
  if(pq.empty()){
    return false;
  }

  std::pair<int, std::pair<int,int>> current_pair = pq.top();
  pq.pop();

  std::pair<int,int> current_cell = current_pair.second;
  int row = current_cell.first;
  int col = current_cell.second;

  if(grid.get_cell(row,col).check_if_visited()){
    return true;
  }

  grid.set_cell_as_visited(row, col);
  grid.set_highlighted_cell(row, col);

  if(grid.get_cell(row,col).is_exit()){
    found_exit = true;
    return false;
  }

  auto neighbors = grid.get_neighbors_search(row, col);

  for(const auto& neighbor: neighbors){
    int n_row = neighbor.first;
    int n_col = neighbor.second;
    int n_g = g_scores[row][col] + 1;

    if(!grid.get_cell(n_row, n_col).check_if_visited() && n_g < g_scores[n_row][n_col]){
      grid.get_cell(n_row, n_col).set_direction(row, col);
      g_scores[n_row][n_col] = n_g;
      int h = manhattan_distance(n_row, n_col, GRID_HEIGHT-1, GRID_WIDTH-1);
      int f = h + n_g;
      pq.push({f,{n_row,n_col}});
    }
  }
  return true;
}

void a_star_animation(Grid& grid) {
  sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "A* Search");

  std::priority_queue<std::pair<int, std::pair<int, int>>, std::vector<std::pair<int, std::pair<int, int>>>, CompareAStarCell> pq;

  // store the values of g and f initialized with a great value
  std::vector<std::vector<int>> g_scores(GRID_HEIGHT, std::vector<int>(GRID_WIDTH, 1000000));
  bool found_exit = false;

  int start_row = 0;
  int start_col = 0;
  int h_start = manhattan_distance(start_row, start_col, GRID_HEIGHT - 1, GRID_WIDTH - 1);
  g_scores[start_row][start_col] = 0;

  pq.push({g_scores[start_row][start_col] + h_start, {start_row, start_col}});

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed)
        window.close();
    }

    window.clear();

    if(!found_exit){
      a_star_animation_step(grid, pq ,g_scores,found_exit);
      sf::sleep(sf::microseconds(15000));
    } else {
      grid.set_highlighted_cell(-1, -1);
      grid.draw_path(window);
      sf::sleep(sf::microseconds(20));
    }

    grid.draw(window);
    window.display();
  }
}
