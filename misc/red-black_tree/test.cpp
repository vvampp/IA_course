#include "red-black_tree.hpp"
#include <iostream>

int main(){
  RedBlackTree tree = RedBlackTree();
  tree.insert(10);
  tree.insert(5);
  tree.insert(15);
  tree.insert(3);
  tree.insert(2);
  tree.insert(2);
  tree.insert(4);
  tree.insert(20);
  tree.insert(16);
  std::cout << "Insertion process finished" << std::endl;
}
