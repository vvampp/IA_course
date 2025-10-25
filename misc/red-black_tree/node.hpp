#ifndef NODE_HPP
#define NODE_HPP

enum Color {
  RED,
  BLACK,
};

class Node {
public:
  int data;
  Color color;
  Node* left;
  Node* right;
  Node* parent;

  Node();
  Node(int data);
};

#endif // !NODE
