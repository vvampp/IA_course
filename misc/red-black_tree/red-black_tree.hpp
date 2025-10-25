#ifndef RED_BLACK_TREE_HPP
#define RED_BLACK_TREE_HPP
#include "node.hpp"

class RedBlackTree{
private:
  Node NIL;
  Node* root;
  void rotate_left(Node* x);
  void rotate_right(Node* y);
  void fix_insert(Node* z);

public:
  RedBlackTree();
  void insert(int data);
};

#endif // !RED_BLACK_TREE_HPP
