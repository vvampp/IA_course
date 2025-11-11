#ifndef BOARD_HPP
#define BOARD_HPP

#include "CONSTANTS.HPP"
#include "SFML/Graphics.hpp"
#include "cell.hpp"
#include <array>
#include <vector>

enum GameState { RUNNING, X_WINS, O_WINS, DRAW };

class Board {
  private:
    std::array<std::array<Cell, BOARD_SIZE>, BOARD_SIZE> grid;
    CellState currentPlayer;
    GameState gameState;
    int movesMade;

    void drawGrid(sf::RenderWindow &window) const;
    GameState checkWin();

  public:
    Board();

    void setCellState(int row, int col, CellState state);
    void incrementMoves();
    GameState evaluateGame();

    bool handleClick(float mouseX, float mouseY);
    void draw(sf::RenderWindow &window) const;
    GameState getGameState() const;
    void reset();
    std::vector<std::pair<int, int>> getMoves() const;
    Board calculateMove(std::pair<int, int> move, CellState player);
};

#endif
