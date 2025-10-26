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
  Node* search(Node*, int);
  Node* minimum(Node*);
  void transplant(Node*, Node*);
  void fix_delete(Node*);

public:
  RedBlackTree();
  Node* get_root();
  void insert(int);
  void delete_node(int);
  void preorder(Node*);
};


#endif // !RED_BLACK_TREE_HPP
