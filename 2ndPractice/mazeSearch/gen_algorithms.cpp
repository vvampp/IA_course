#include "gen_algorithms.hpp"

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Window/VideoMode.hpp>
#include <iostream>

#include "CONSTANTS.HPP"
#include "utils.hpp"

Grid prim_maze_animation(unsigned int seed) {
  srand(seed);
  Grid grid;
  std::vector<std::tuple<int, int, int, int>> walls;

  int start_row = get_random_number(GRID_HEIGHT);
  int start_col = get_random_number(GRID_WIDTH);
  grid.set_cell_as_visited(start_row, start_col);

  if (start_row > 0)
    walls.push_back(std::make_tuple(start_row, start_col, start_row - 1, start_col));
  if (start_row < GRID_HEIGHT - 1)
    walls.push_back(std::make_tuple(start_row, start_col, start_row + 1, start_col));
  if (start_col > 0)
    walls.push_back(std::make_tuple(start_row, start_col, start_row, start_col - 1));
  if (start_col < GRID_WIDTH - 1)
    walls.push_back(std::make_tuple(start_row, start_col, start_row, start_col + 1));

  sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Generation...");
  bool animation_completed = false;

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed)
        window.close();
    }
    if (!animation_completed)
      animation_completed = !grid.prim_maze_animation_step(walls);

    grid.draw(window);
    window.display();
    window.clear();

    sf::sleep(sf::milliseconds(1));
  }

  grid.reset_visits();
  grid.set_entry(0, 0);
  grid.set_exit(GRID_HEIGHT - 1, GRID_WIDTH - 1);

  return grid;
}
