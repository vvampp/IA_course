#include <SFML/Window/Keyboard.hpp>
#include <algorithm>
#include <iostream>
#include <ostream>

#include "CONSTANTS.HPP"
#include "gen_algorithms.hpp"
#include "search_algorithms.hpp"

int main() {
  sf::Font font;
  if (!font.loadFromFile("./fonts/arial.ttf")) {
    return -1;
  }

  Grid maze;
  unsigned int seed = time(0);

  // Maze generation step (using prim)
  maze = prim_maze_animation(seed);

  // Search selection
  std::cout << "Select an algorithm: " << std::endl;
  std::cout << "0. DFS\t1. BFS\t2. A*" << std::endl;

  int algorithm_choice;
  std::cin >> algorithm_choice;

  switch (algorithm_choice) {
    case 0:
      dfs_animation(maze);
      break;
    case 1:
      bfs_animation(maze);
      break;
    case 2:
      // implement A* over grid
      break;
  }
}
