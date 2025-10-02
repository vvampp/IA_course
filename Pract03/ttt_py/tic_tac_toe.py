import turtle 
import sys
import copy
from enum import Enum

WIDTH = 600
HEIGHT = 600 
BOARD_SIZE = 3
LINE_THICKNESS = 6
SQUARE_SIZE = 120
BOARD_LENGHT = BOARD_SIZE * SQUARE_SIZE

PADDING_X = -BOARD_LENGHT / 2
PADDING_Y = -BOARD_LENGHT / 2

# Cell class implementation
class CellState(Enum):
    EMPTY = 0
    X_PLAYER = 1
    O_PLAYER = 2

class Cell:
    def __init__(self,x=0.0,y=0.0,size=0.0):
        self.state = CellState.EMPTY
        self.position = (x,y)
        self.size = size

    def set_state(self, new_state: CellState):
        self.state = new_state

    def get_state(self) -> CellState:
        return self.state

    def draw(self, pen: turtle.Turtle):
        padding = 0.15 * self.size
        draw_x = self.position[0] + padding
        draw_y = self.position[1] + padding
        inner_size = self.size - 2 * padding

        pen.width(LINE_THICKNESS)

        if self.state == CellState.X_PLAYER:
            pen.penup()
            pen.goto(draw_x,draw_y)
            pen.pendown()
            pen.goto(draw_x + inner_size, draw_y + inner_size)

            pen.penup()
            pen.goto(draw_x, draw_y + inner_size)
            pen.pendown()
            pen.goto(draw_x + inner_size, draw_y)

        elif self.state == CellState.O_PLAYER:
            pen.penup()
            pen.goto(draw_x + inner_size/2, draw_y)
            pen.pendown()
            pen.circle(inner_size/2)

# Board class implementation
class GameState(Enum):
    RUNNING = 0
    X_WINS = 1
    O_WINS = 2
    DRAW = 3

class Board:
    def __init__(self, pen: turtle.Turtle, window: turtle.Screen):
        self.pen = pen
        self.window = window
        self.current_player = CellState.X_PLAYER 
        self.game_state = GameState.RUNNING
        self.moves_made = 0

        self.grid = [
                [
                    Cell(
                        x = col * SQUARE_SIZE + PADDING_X,
                        y = row * SQUARE_SIZE + PADDING_Y,
                        size = SQUARE_SIZE
                        )
                    for col in range(BOARD_SIZE)
                    ]
                for row in range(BOARD_SIZE)
                ]

    def set_cell_state(self, row, col, state):
        self.grid[row][col].set_state(state)

    def increment_moves(self):
        self.moves_made += 1

    def evaluate_game(self) -> GameState:
        self.game_state = self.check_win()

    def draw(self, pen: turtle.Turtle):
        pen.width(LINE_THICKNESS)
        for i in range(1,BOARD_SIZE):
            pen.penup()
            pen.goto(PADDING_X + i * SQUARE_SIZE, PADDING_Y)
            pen.pendown()
            pen.goto(PADDING_X + i * SQUARE_SIZE, PADDING_Y + BOARD_LENGHT)
            pen.penup()
            pen.goto(PADDING_X, PADDING_Y + i * SQUARE_SIZE)
            pen.pendown()
            pen.goto(PADDING_X + BOARD_LENGHT, PADDING_Y + i * SQUARE_SIZE)

        for row in self.grid:
            for cell in row:
                cell.draw(pen)

    def check_win(self):
        def _check_player(player: CellState) -> bool:
            for i in range(BOARD_SIZE):
                row_win = all(self.grid[i][j].get_state() == player for j in range(BOARD_SIZE))
                col_win = all(self.grid[j][i].get_state() == player for j in range(BOARD_SIZE))
                if row_win or col_win:
                    return True

            diag1_win = all(self.grid[i][i].get_state() == player for i in range(BOARD_SIZE))
            diag2_win = all(self.grid[i][BOARD_SIZE-1-i].get_state() == player for i in range(BOARD_SIZE))

            return diag1_win or diag2_win

        if _check_player(CellState.X_PLAYER):
            return GameState.X_WINS
        if _check_player(CellState.O_PLAYER):
            return GameState.O_WINS
        if self.moves_made == BOARD_SIZE * BOARD_SIZE:
            return GameState.DRAW
        return GameState.RUNNING

    def get_moves(self) -> list[tuple[int,int]]:
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.grid[r][c].get_state() == CellState.EMPTY:
                    moves.append((r,c))
        return moves
    
    def calculate_move(self, move: tuple[int,int], player: CellState):
        board_copy = Board(pen=None,window=None)
        board_copy.grid = copy.deepcopy(self.grid)
        board_copy.moves_made = self.moves_made 
        board_copy.current_player = self.current_player

        board_copy.set_cell_state(move[0],move[1],player)
        board_copy.increment_moves()
        board_copy.evaluate_game()
        return board_copy

    def handle_click(self, mouse_x: float, mouse_y: float) -> bool:
        if self.game_state != GameState.RUNNING:
            return False

        col = int((mouse_x - PADDING_X) / SQUARE_SIZE)
        row = int((mouse_y - PADDING_Y) / SQUARE_SIZE)

        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            if self.grid[row][col].get_state() == CellState.EMPTY:
                self.set_cell_state(row, col, self.current_player)
                self.moves_made += 1
                self.game_state = self.check_win()


                if self.game_state == GameState.RUNNING:
                    self.current_player = CellState.O_PLAYER

                    make_move(self)

                    self.game_state = self.check_win()

                if self.game_state == GameState.RUNNING:
                    self.current_player = CellState.X_PLAYER

                # Two human players
                #if self.game_state == GameState.RUNNING:
                #    if self.current_player == CellState.X_PLAYER:
                #        self.current_player = CellState.O_PLAYER
                #    else:
                #        self.current_player = CellState.X_PLAYER

                self.pen.clear()
                self.draw(self.pen)
                self.window.update()

                if self.game_state == GameState.X_WINS:
                    print("X Player wins!")
                if self.game_state == GameState.O_WINS:
                    print("O Player wins!")
                if self.game_state == GameState.DRAW:
                    print("Game Over, it's a draw...")

    def get_game_state(self) -> GameState:
        return self.game_state

    def reset(self):
        self.current_player = CellState.X_PLAYER
        self.game_state = GameState.RUNNING
        self.moves_made = 0
        for row in self.grid:
            for cell in row:
                cell.set_state(CellState.EMPTY)


def utility(board: Board) -> int:
    game_state = board.get_game_state()
    if game_state == GameState.X_WINS:
        return -1
    if game_state == GameState.DRAW:
        return 0
    if game_state == GameState.O_WINS:
        return 1
    else:
        return 0

def is_terminal_state(board: Board) -> bool:
    return not board.get_moves() or board.get_game_state() != GameState.RUNNING

def alpha_beta_minmax(board: Board, depth: int, player: CellState, alpha: int, beta: int) -> tuple[int,tuple[int,int]]:
    if(is_terminal_state(board) or depth == 0):
        return (utility(board), (-1,-1))

    best_move = (-1,-1)

    if player == CellState.O_PLAYER:
        max_eval = -sys.maxsize
        for move in board.get_moves():
            new_board = board.calculate_move(move,CellState.O_PLAYER)
            eval,_ = alpha_beta_minmax(new_board, depth-1, CellState.X_PLAYER, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha,eval) 
            if alpha >= beta:
                break
        return (max_eval,best_move)
    else:
        min_eval = sys.maxsize
        for move in board.get_moves():
            new_board = board.calculate_move(move,CellState.X_PLAYER)
            eval,_ = alpha_beta_minmax(new_board,depth-1, CellState.O_PLAYER, alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta,eval)
            if beta <= alpha:
                break
        return (min_eval, best_move)


def make_move(board: Board):
    result = alpha_beta_minmax(board,6,CellState.O_PLAYER,-sys.maxsize,sys.maxsize)
    move = result[1]
    if move[0] != -1 and move[1] != -1:
        board.set_cell_state(move[0], move[1], CellState.O_PLAYER)
        board.increment_moves()
        board.evaluate_game()

def main():
    window = turtle.Screen()
    window.setup(WIDTH,HEIGHT)
    window.title("Tic Tac Toe")
    window.tracer(0)

    pen = turtle.Turtle()
    pen.hideturtle()
    pen.speed(0)

    game_board = Board(pen,window)

    def restart_game():
        game_board.reset()
        pen.clear()
        game_board.draw(pen)
        window.update()

    window.onscreenclick(game_board.handle_click)

    window.listen()
    window.onkey(restart_game, 'r')

    game_board.draw(pen)
    window.update()

    turtle.done()

if __name__ == "__main__":
    main()
