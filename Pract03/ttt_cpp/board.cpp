#include "board.hpp"
#include <iostream>

Board::Board() : currentPlayer(X_PLAYER), gameState(RUNNING), movesMade(0) {
  // Grid initialization
  for (int i = 0 ; i < BOARD_SIZE ; ++i)
    for (int j = 0 ; j < BOARD_SIZE; ++j){
      float x = (float)j * SQUARE_SIZE;
      float y = (float)i * SQUARE_SIZE;
      grid[i][j] = Cell(x,y,SQUARE_SIZE);
    }
}

void Board::drawGrid(sf::RenderWindow& window) const {
  const float board_length = (float) BOARD_SIZE * SQUARE_SIZE;

  for (int j = 1 ; j < BOARD_SIZE ; ++j){
    sf::RectangleShape line(sf::Vector2f(LINE_THICKNESS, board_length));
    line.setFillColor(sf::Color::White);
    line.setPosition((float)j * SQUARE_SIZE - LINE_THICKNESS / 2.0f, 0.0f);
    window.draw(line);
  }

  for (int i = 1; i < BOARD_SIZE; ++i){
    sf::RectangleShape line(sf::Vector2f(board_length,LINE_THICKNESS));
    line.setFillColor(sf::Color::White);
    line.setPosition(0.0f, (float)i * SQUARE_SIZE - LINE_THICKNESS / 2.0f);
    window.draw(line);
  }
}

GameState Board::checkWin(){
  // Helper lambda
  auto check = [&](CellState player){
    // Check rows / columns
    for(int i = 0; i < BOARD_SIZE ; ++i){
      bool row_win = true;
      bool col_win = true;
      for(int j = 0 ; j < BOARD_SIZE; ++j){
        if(grid[i][j].getState() != player) row_win = false;
        if(grid[j][i].getState() != player) col_win = false;
      }
      if(row_win || col_win) return true;
    }

    bool diag_win1 = true;
    // Check diagonal (top-left -> bottom-right)
    for (int i = 0; i < BOARD_SIZE ; ++i){
      if(grid[i][i].getState() != player){
        diag_win1 = false;
        break;
      }
    }
    // Check anti diagonal (top-right -> bottom-left)
    bool diag_win2 = true;
    for (int i = 0; i < BOARD_SIZE ; ++i){
      if(grid[i][BOARD_SIZE-1-i].getState() != player){
        diag_win2 = false;
        break;
      }
    }
    if (diag_win1 || diag_win2) return true;
    return false;
  };

  if(check(X_PLAYER)) return X_WINS;
  if(check(O_PLAYER)) return O_WINS;
  if(movesMade == BOARD_SIZE * BOARD_SIZE) return DRAW;
  return RUNNING;
}

bool Board::handleClick(float mouseX, float mouseY){
  if(gameState != RUNNING)
    return false;
  
  int col = (int)(mouseX/SQUARE_SIZE);
  int row = (int)(mouseY/SQUARE_SIZE);

  if(row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE){
    if(grid[row][col].getState() == EMPTY){
      grid[row][col].setState(currentPlayer);
      movesMade++;

      gameState = checkWin();

      if(gameState == RUNNING){
        currentPlayer = (currentPlayer == X_PLAYER)? O_PLAYER : X_PLAYER;
      }
      return true;
    }
  }
  return false;
}

void Board::draw(sf::RenderWindow& window) const {
  drawGrid(window);
  for(int i = 0; i < BOARD_SIZE; ++i){
    for(int j = 0; j < BOARD_SIZE; ++j){
      grid[i][j].draw(window);
    }
  }
}

GameState Board::getGameState() const {
  return gameState;
}

void Board::reset(){
  for(int i = 0 ; i < BOARD_SIZE; ++i)
    for(int j = 0 ; j < BOARD_SIZE; ++j)
      grid[i][j].setState(EMPTY);
  currentPlayer = X_PLAYER;
  gameState = RUNNING;
  movesMade = 0;
}
