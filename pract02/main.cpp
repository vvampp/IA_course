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
  sf::Text search_instructions;
  search_instructions.setString("J. DFS\nK. BFS\nL. A*\nQ. Salir");
  search_instructions.setCharacterSize(25);
  search_instructions.setOrigin(search_instructions.getGlobalBounds().width / 2, search_instructions.getGlobalBounds().height / 2);
  search_instructions.setPosition(WIDTH / 2 + 50, HEIGHT / 2 + 20);

  Grid maze;
  unsigned int seed = time(0);

  // Maze generation step (using prim)
  maze = prim_maze_animation(seed);

  // DEBUG: Print wall states for your 3x3 example
  std::cout << "\n=== WALL STATES DEBUG ===" << std::endl;
  for (int i = 0; i < GRID_HEIGHT; i++) {
    for (int j = 0; j < GRID_WIDTH; j++) {
      std::cout << "Cell (" << i << "," << j << "): ";
      std::cout << "T=" << maze.get_cell(i, j).has_top_wall() << " ";
      std::cout << "R=" << maze.get_cell(i, j).has_right_wall() << " ";
      std::cout << "B=" << maze.get_cell(i, j).has_bottom_wall() << " ";
      std::cout << "L=" << maze.get_cell(i, j).has_left_wall() << std::endl;
    }
  }

  std::cout << "Select an algorithm: " << std::endl;
  std::cout << "0. DFS\t1. BFS\t2. A*" << std::endl;

  int algorithm_choice;
  std::cin >> algorithm_choice;

  switch (algorithm_choice) {
    case 0:
      dfs_animation(maze);
      break;
    case 1:
      // implemnet BFS over grid
      break;
    case 2:
      // implement A* over grid
      break;
  }
}
