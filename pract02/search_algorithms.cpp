#include "search_algorithms.hpp"

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/System/Sleep.hpp>
#include <SFML/System/Time.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Window/VideoMode.hpp>
#include <iostream>
#include "CONSTANTS.HPP"

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

    if (!found_exit){
      dfs_animation_step(grid, stack, found_exit);
      sf::sleep(sf::microseconds(15000));
    }
    else{
      grid.set_highlighted_cell(-1, -1);
      grid.draw_path(window);
      sf::sleep(sf::microseconds(20));
    }

    grid.draw(window);
    window.display();

  }
}
