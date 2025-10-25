#ifndef RED_BLACK_TREE_HPP
#define RED_BLACK_TREE_HPP
#include "node.hpp"

class RedBlackTree{
public:
  RedBlackTree();
  Node NIL;
  Node* root;
  void rotate_left(Node* x);
  void rotate_right(Node* y);
  void insert(int data);
  void fix_insert(Node* z);
};

#endif // !RED_BLACK_TREE_HPP
