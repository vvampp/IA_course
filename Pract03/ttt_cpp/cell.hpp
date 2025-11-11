#ifndef CELL_HPP
#define CELL_HPP
#include <SFML/Graphics.hpp>

enum CellState {
    EMPTY,
    X_PLAYER,
    O_PLAYER,
};

class Cell {
  private:
    CellState state;
    sf::Vector2f position;
    float size;

  public:
    Cell();
    Cell(float x, float y, float s);

    void setState(CellState state);
    CellState getState() const;

    void draw(sf::RenderWindow &window) const;
};

#endif
