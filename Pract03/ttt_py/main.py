import turtle 
from enum import Enum

BOARD_SIZE = 3
LINE_THICKNESS = 6

class CellState(enum):
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

        elif self.state == CellState.O_PLAYER
            pen.penup()
            pen.goto(draw_x + inner_size/2, draw_y)
            pen.pendown()
            pen.circle(inner_size/2)


def get_coordinates(x,y):
    print(f'Clicked over x={x}, y={y}')


def main():
    window = turtle.Screen()
    window.title("Tic Tac Toe")
    window.onscreenclick(get_coordinates)
    turtle.done()


if __name__ == "__main__":
    main()
