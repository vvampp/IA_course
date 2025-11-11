#include <SFML/Window/Keyboard.hpp>
#include <iostream>

#include "CONSTANTS.HPP"
#include "gen_algorithms.hpp"
#include "search_algorithms.hpp"

// Forward declarations
void dfs_animation(Grid &grid);
void bfs_animation(Grid &grid);
void a_star_animation(Grid &grid);

int main() {
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Maze Solver");

    Grid maze;
    unsigned int seed = time(0);

    // Maze generation step (using prim)
    maze = prim_maze_animation(seed);

    // Main program loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            if (event.type == sf::Event::KeyPressed) {
                switch (event.key.code) {
                case sf::Keyboard::J:
                    maze.reset_visits();
                    dfs_animation(maze);
                    break;
                case sf::Keyboard::K:
                    maze.reset_visits();
                    bfs_animation(maze);
                    break;
                case sf::Keyboard::L:
                    maze.reset_visits();
                    a_star_animation(maze);
                    break;
                case sf::Keyboard::Q:
                    window.close();
                    break;
                default:
                    break;
                }
            }
        }

        window.clear();
        maze.draw(window);
        window.display();
    }

    return 0;
}
