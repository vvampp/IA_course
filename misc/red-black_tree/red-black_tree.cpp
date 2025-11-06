#include "red-black_tree.hpp"
#include "node.hpp"
#include <iostream>
#include <string>

RedBlackTree::RedBlackTree() : NIL(Node()), root(&NIL) {}

Node* RedBlackTree::get_root(){
  return root;
}
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
  new_node->parent = parent;
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

Node* RedBlackTree::search(Node* node, int key){
  if(node == &NIL || key == node->data)
    return node;
  // recursively call search function checking for values and traversing nodes
  if(key < node->data)
    return search(node->left,key);
  else
    return search(node->right,key);
}

Node* RedBlackTree::minimum(Node* node){
  // traverse left childs until a leaf
  while(node->left != &NIL){
    node = node->left;
  }
  return node;
}

void RedBlackTree::transplant(Node* u, Node* v){
  // if u is root, make v root
  if(u->parent == nullptr)
    root = v;
  // if u is left child of its parent, make v left child of former u's parent
  else if(u==u->parent->left)
    u->parent->left = v;
  // u is right child of its parent, make v right child of former u's parent
  else
    u->parent->right = v;
  // associate u's parent to v's parent
  v->parent = u->parent;
}

void RedBlackTree::delete_node(int data){
  Node* z = search(root,data);
  if(z==&NIL)
    return;

  Node* y = z;
  Color y_original_color = y->color;

  Node* x;
  if(z->left == &NIL){
    x = z->right;
    transplant(z, z->right);
  }
  else if(z->right == &NIL){
    x = z->left;
    transplant(z, z->left);
  }
  // parent node has two children
  else {
    y = minimum(z->right);
    y_original_color = y->color;
    x = y->right;
    // if y's parent is the target node, update x's parent to y
    if (y->parent == z)
      x->parent = y;
    else{
      transplant(y,y->right);
      y->right = z->right;
      y->right->parent = y;
    }

    transplant(z, y);
    y->left = z->left;
    y->left->parent = y;
    y->color = z->color;
  }

  if(y_original_color == Color::BLACK)
    fix_delete(x);
}

void RedBlackTree::fix_delete(Node* x){
  // uncle node
  Node* w;
  while(x != root && x->color == Color::BLACK){
    // replaced node is left node
    if(x == x->parent->left){
      w = x->parent->right;
      // if uncle is RED
      if(w->color == Color::RED){
        w->color = Color::BLACK;
        x->parent->color = Color::RED;
        rotate_left(x->parent);
        // reasignment of uncle after the rotation
        w = x->parent->right;
      }
      // if the uncle and its children are all black
      if(w->left->color == Color::BLACK && w->right->color == Color::BLACK){
        w->color = Color::RED;
        x = x->parent;
      }
      // uncle has one red child
      else{
        // left is the red child
        if(w->right->color == Color::BLACK){
          w->left->color = Color::BLACK;
          w->color = Color::RED;
          rotate_right(w);
          w = x->parent->right;
        }
        w->color = x->parent->color;
        x->parent->color = Color::BLACK;
        w->right->color = Color::BLACK;
        rotate_left(x->parent);
        x = root;
      }
    }
    else {
      w = x->parent->left;

      if(w->color == Color::RED){
        w->color = Color::BLACK;
        x->parent->color = Color::RED;
        rotate_right(x->parent);
        w = x->parent->left;
      }

      if(w->right->color == Color::BLACK && w->left->color == Color::BLACK){
        w->color = Color::RED;
        x = x->parent;
      }
      else{
        // right child is red
        if(w->left->color == Color::BLACK){
          w->right->color = Color::BLACK;
          w->color = Color::RED;
          rotate_left(w);
          w = x->parent->left;
        }
        w->color = x->parent->color;
        x->parent->color = Color::BLACK;
        w->left->color = Color::BLACK;
        rotate_right(x->parent);
        x = root;
      }
    }
  }
  x->color = Color::BLACK;
}


void RedBlackTree::preorder(Node* node){
  if(node == &NIL){
    return;
  }

  std::string node_color;
  node->color == Color::BLACK ? node_color = "BLACK" : node_color = "RED";

  std::cout<< node->data << " : " << node_color << std::endl;
  preorder(node->left);
  preorder(node->right);
}

