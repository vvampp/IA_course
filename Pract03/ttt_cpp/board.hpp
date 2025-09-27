#ifndef BOARD_HPP
#define BOARD_HPP

#include "SFML/Graphics.hpp"
#include "CONSTANTS.HPP"
#include "cell.hpp"
#include <array>

enum GameState {
  RUNNING,
  X_WINS,
  O_WINS,
  DRAW
};

class Board{
  private:
    std::array<std::array<Cell, BOARD_SIZE>, BOARD_SIZE> grid;
    CellState currentPlayer;
    GameState gameState;
    int movesMade;

    void drawGrid (sf::RenderWindow& window) const;
    GameState checkWin();

  public:
    Board();
    bool handleClick(float mouseX, float mouseY);
    void draw(sf::RenderWindow& window) const;
    GameState getGameState() const;
    void reset();
};

#endif
