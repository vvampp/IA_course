#include "alpha_beta_minmax.hpp"

int utility(const Board& board){
  GameState gameState = board.getGameState();
  switch (gameState) {
    case(X_WINS):
      return -1;
    case(DRAW):
      return 0;
    case(O_WINS):
      return 1;
    default:
      return 0;
  }
}

bool isTerminalState(const Board& board){
  return (board.getMoves().empty() || board.getGameState() != RUNNING);
}

std::pair<int,std::pair<int,int>> alpha_beta_minmax(Board board, int depth, CellState player, int alpha, int beta){
  if(isTerminalState(board) || depth == 0)
    return {utility(board),{-1,-1}};

  std::pair<int,int> bestMove = {-1,-1};

  // Maximazing for O
  if(player == O_PLAYER){
    int max_eval = INT_MIN;
    for(auto& move : board.getMoves()){
      Board newBoard = board.calculateMove(move,O_PLAYER);
      int eval = alpha_beta_minmax(newBoard,depth-1, X_PLAYER, alpha, beta).first;
      if(eval > max_eval){
        max_eval = eval;
        bestMove = move;
      }
      alpha = std::max(alpha,eval);
      if(alpha >= beta) break;
    }
    return {max_eval,bestMove};
    
  } else { // Minimizing for X
    int min_eval = INT_MAX;
    for(auto& move : board.getMoves()){
      Board newBoard = board.calculateMove(move,X_PLAYER);
      int eval = alpha_beta_minmax(newBoard,depth-1, O_PLAYER, alpha, beta).first;
      if(eval < min_eval){
        min_eval = eval;
        bestMove = move;
      }
      beta = std::min(beta,eval);
      if (beta <= alpha) break;
    }
    return {min_eval,bestMove};
  }
}

void makeMove(Board& board){
  auto result = alpha_beta_minmax(board,4,O_PLAYER,INT_MIN,INT_MAX);
  std::pair<int,int> move = result.second;
  if(move.first != -1 && move.second != -1){
    // board.grid[move.first][move.second].setState(O_PLAYER);
    board.setCellState(move.first,move.second,O_PLAYER);
    // board.movesMade++;
    board.incrementMoves();
    // board.gameState = board.checkWin();
    board.evaluateGame();
  }
}

