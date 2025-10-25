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
  Node* get_root();
  void insert(int data);
  Node* search(Node* node, int key);
  Node* minimum(Node* node);
  void transplant(Node* u, Node* v);
};


#endif // !RED_BLACK_TREE_HPP
