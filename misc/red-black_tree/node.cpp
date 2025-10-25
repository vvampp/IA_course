#include "node.hpp"
#include <iostream>

Node::Node()
  : data(0), color(Color::BLACK), left(nullptr), right(nullptr), parent(nullptr) {}
Node::Node(int data)
  : data(data), color(Color::RED), left(nullptr), right(nullptr), parent(nullptr) {}
