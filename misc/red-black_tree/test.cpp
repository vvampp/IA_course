#include "red-black_tree.hpp"
#include <iostream>

int main(){
  RedBlackTree tree = RedBlackTree();
  tree.insert(1);
  tree.insert(2);
  tree.insert(3);
  tree.insert(4);
  tree.insert(5);
  tree.delete_node(1);
  std::cout << "Insertion process finished" << std::endl;
  Node * root = tree.get_root();
  std::cout << "Value at tree's root: " << root->data << std::endl;
}
