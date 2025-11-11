#include "cell.hpp"
#include "CONSTANTS.HPP"

Cell::Cell() : state(EMPTY), position(0.0f, 0.0f), size(0.0f) {
    // Empty initialization complete
}

Cell::Cell(float x, float y, float s) : state(EMPTY), position(x, y), size(s) {
    // Initialization complete ...
}

void Cell::setState(CellState newState) { state = newState; }

CellState Cell::getState() const { return state; }

void Cell::draw(sf::RenderWindow &window) const {
    const float padding = 0.15f * size;
    const float inner_size = size - 2 * padding;
    const float thickness = LINE_THICKNESS;
    const sf::Vector2f draw_pos = position + sf::Vector2f(padding, padding);

    if (state == X_PLAYER) {

        sf::RectangleShape X_line1(sf::Vector2f(inner_size * 1.4f, thickness));
        X_line1.setFillColor(sf::Color::White);
        X_line1.setPosition(draw_pos);
        X_line1.rotate(45.0f);
        window.draw(X_line1);

        sf::RectangleShape X_line2(sf::Vector2f(inner_size * 1.4f, thickness));
        X_line2.setFillColor(sf::Color::White);
        X_line2.setPosition(draw_pos.x, draw_pos.y + inner_size);
        X_line2.rotate(-45.0f);
        window.draw(X_line2);

    } else if (state == O_PLAYER) {

        sf::CircleShape circle(inner_size / 2.0f);
        circle.setFillColor(sf::Color::Transparent);
        circle.setOutlineThickness(thickness);
        circle.setOutlineColor(sf::Color::White);
        circle.setPosition(draw_pos);
        window.draw(circle);
    }
}
