#include "CONSTANTS.HPP"
#include "board.hpp"
#include <SFML/Graphics.hpp>
#include <iostream>

int main() {
  sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Tic-Tac-Toe!");
  window.setFramerateLimit(60);

  Board gameBoard;

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed) {
        window.close();
      }

      if (event.type == sf::Event::MouseButtonPressed) {
        if (event.mouseButton.button == sf::Mouse::Left) {
          float mouseX = (float)event.mouseButton.x;
          float mouseY = (float)event.mouseButton.y;

          bool moveMade = gameBoard.handleClick(mouseX, mouseY);

          if (moveMade) {
            GameState state = gameBoard.getGameState();
            if (state != RUNNING) {
              std::cout << "Game Over! ";
              if (state == X_WINS)
                std::cout << "X Wins!\n";
              else if (state == O_WINS)
                std::cout << "O Wins!\n";
              else if (state == DRAW)
                std::cout << "It's a Draw!\n";
            }
          }
        }
      }

      if (event.type == sf::Event::KeyPressed) {
        if (event.key.code == sf::Keyboard::R) {
          gameBoard.reset();
          std::cout << "Game Reset.\n";
        }
      }
    }

    window.clear(sf::Color::Black);
    gameBoard.draw(window);
    window.display();
  }

  return 0;
}
