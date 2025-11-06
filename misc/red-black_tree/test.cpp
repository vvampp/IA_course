#include "red-black_tree.hpp"
#include <iostream>

int main(){
  RedBlackTree tree = RedBlackTree();

  tree.insert(32);
  tree.insert(2);
  tree.insert(98);
  tree.insert(97);
  tree.insert(54);
  tree.insert(43);

  tree.delete_node(32);

  tree.preorder(tree.get_root());
}
