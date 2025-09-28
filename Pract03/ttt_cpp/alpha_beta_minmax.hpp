#ifndef ALPHA_BETA_MINMAX
#define ALPHA_BETA_MINMAX

#include "cell.hpp"
#include "board.hpp"
#include <climits>

int utility(const Board& board);
bool isTerminalState(const Board& board);

std::pair<int,std::pair<int,int>> alpha_beta_minmax(Board board, int depth, CellState player, int alpha, int beta);

void makeMove(Board& board);

#endif 
