#include "red-black_tree.hpp"

RedBlackTree::RedBlackTree() : NIL(Node()), root(&NIL) {}

void RedBlackTree::rotate_left(Node* x){

  // y is defined as the rigth child of x
  Node* y = x->right;

  // x's right child is now y's left child
  x->right = y->left;

  // if y has a left child, make x the parent of this node
  if( y->left != &NIL)
    y->left->parent = x;
  
  // make x's parent y's parent now
  y->parent = x->parent;

  // if x was the root, make y the root
  if (x->parent == nullptr)
    root = y;
  // if x is the left child of it's father, make y x's father left child
  else if(x == x->parent->left)
    x->parent->left = y;
  // if x is the right child of it's father, make y x's father right child
  else
    x->parent->right = y;

  // make x y's left child
  y->left = x;

  // make y x's parent
  x->parent = y;
}

void RedBlackTree::rotate_right(Node *y){

  // x is defined as the left child of y
  Node* x = y->left;

  // y's left child is now x's right child
  y->left = x->right;

  // if x was the root, make y the root
  if(x->right == &NIL)
    x->right->parent = y;

  // make y's parent x's parent
  x->parent = y->parent;

  // if y was the root make x the root
  if(y->parent ==  nullptr)
    root = x;
  // if y was the right child of it's parent, make x the right child of y's parent
  else if(y == y->parent->right)
    y->parent->right = x;
  // if y was the left child of it's parent, make x the left child of y's parent
  else
    y->parent->left = x;

  // make y x's right child
  x->right = y;

  // make x y's parent
  y->parent = x;
}

void RedBlackTree::insert(int data){
  Node * new_node = new Node(data);

  // initialize child nodes as NIL
  new_node->left = &NIL;
  new_node->right = &NIL;

  // tracking traversal
  Node* parent = nullptr;
  Node* current = root;

  // traverse tree checking for lower/greater conditions
  while(current != &NIL){
    parent = current;

    if(new_node->data < current->data) 
      current = current->left;
    else
      current = current->right;
  }

  // set parent once the traversal is done
  new_node->parent = current->parent;
  // check if root
  if (new_node->parent == nullptr)
    root = new_node;
  // check for relation between new node and its parent
  else if(new_node->data < parent->data)
    parent->left = new_node;
  else
    parent->right = new_node;

  fix_insert(new_node);
}

void RedBlackTree::fix_insert(Node* z){
  // while the parent is RED
  while(z->parent != nullptr && z->parent->color == Color::RED){
    // if parent is left child of grandparent
    if(z->parent == z->parent->parent->left){
      // uncle (y) is the right sibiling of parent
      Node* y = z->parent->parent->right;
      // check if uncle is RED
      if(y->color == Color::RED){
        z->parent->color = Color::BLACK;
        y->color = Color::BLACK;
        z->parent->parent->color = Color::RED;
        // traverse up
        z = z->parent->parent;
      }
      // uncle is BLACK
      else{
        // node is the right child of its parent
        if(z==z->parent->right){
          z = z->parent;
          rotate_left(z);
        }
        z->parent->color = Color::BLACK;
        z->parent->parent->color = Color::RED;
        rotate_right(z->parent->parent);
      }
    }
    // parent is the right child of the grandparent
    else{
      // uncle (y) is the left sibiling of parent
      Node* y = z->parent->parent->left;
      // check if uncle is RED
      if(y->color == Color::RED){
        z->parent->color = Color::BLACK;
        y->color = Color::BLACK;
        z->parent->parent->color = Color::RED;
        // traverse up
        z = z->parent->parent;
      }
      // uncle is BLACK
      else{
        // node is the right child of its parent
        if(z==z->parent->left){
          z = z->parent;
          rotate_right(z);
        }
        z->parent->color = Color::BLACK;
        z->parent->parent->color = Color::RED;
        rotate_left(z->parent->parent);
      }
    }
  }
  root->color = Color::BLACK;
}


