import numpy as np

class AABB:
    def __init__(self, min_x, min_y, min_z, max_x, max_y, max_z, obj=None):
        self.min_x = min_x
        self.min_y = min_y
        self.min_z = min_z
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z
        self.obj = obj
        self.node = None
    
    def update(self, min_x, min_y, min_z, max_x, max_y, max_z):
        self.min_x = min_x
        self.min_x = min_y
        self.min_x = min_z
        self.max_x = max_x
        self.max_x = max_y
        self.max_x = max_z
        
    def volume(self):
        return (self.max_x - self.min_x) * (self.max_y - self.min_y) * (self.max_z - self.min_z)
    
    def union(aabbs):
        min_x = min(aabbs, key=lambda x: x.min_x).min_x
        min_y = min(aabbs, key=lambda x: x.min_y).min_y
        min_z = min(aabbs, key=lambda x: x.min_z).min_z
        max_x = max(aabbs, key=lambda x: x.max_x).max_x
        max_y = max(aabbs, key=lambda x: x.max_y).max_y
        max_z = max(aabbs, key=lambda x: x.max_z).max_z
        return AABB(min_x, min_y, min_z, max_x, max_y, max_z)
    
    def contains(self, other):
        return (self.min_x <= other.min_x and
                self.max_x >= other.max_x and
                self.min_y <= other.min_y and
                self.max_y >= other.max_y and
                self.min_z <= other.min_z and
                self.max_z >= other.max_z)
    
    def intersects(self, other):
        return (self.min_x <= other.max_x and
                self.max_x >= other.min_x and
                self.min_y <= other.max_y and
                self.max_y >= other.min_y and
                self.min_z <= other.max_z and
                self.max_z >= other.min_z)
        
class AABBNode:
    def __init__(self):
        self.parent = []
        self.children = []
        self.collision_check = False
        self.data = None
        self.aabb = None
        
    def is_leaf(self):
        return not self.children
    
    def set_branch(self, n0, n1):
        n0.parent, n1.parent = [self], [self]
        self.children = [n0, n1]
        
    def set_leaf(self, data):
        self.data, data.node = data, self
        self.children = []
        
    def update_aabb(self):
        if self.is_leaf():
            margin = 0
            self.aabb = AABB(self.data.min_x - margin, self.data.min_y - margin,
                             self.data.min_z - margin, self.data.max_x + margin,
                             self.data.max_y + margin, self.data.max_z + margin,)
        else:
            self.aabb = AABB.union([child.aabb for child in self.children])
            
    def get_sibling(self):
        if self == self.parent[0].children[0]:
            return self.parent[0].children[1]
        else:
            return self.parent[0].children[0]
        
class AABBTree:
    def __init__(self):
        self.root = []
        self.collision_pairs = []
        self.invalid_nodes = []
        self.tree_updated = True
        
    def insert(self, aabb):
        self.tree_updated = True
        # if root exists create a new leaf and insert the node
        if self.root:
            node = AABBNode()
            node.set_leaf(aabb)
            node.update_aabb()
            self.insert_node(node, self.root, index=0)
        # if there is not root make the root as the leaf node
        else:
            self.root = [AABBNode()]
            self.root[0].set_leaf(aabb)
            self.root[0].update_aabb()
            
    def insert_node(self, node, parent, index=0):
        stack = [(node, parent, index)]
        while stack:
            node, parent, index = stack.pop()
            parent_node = parent[index]
            
            # if parent node is leaf, split the node and insert children nodes
            if parent_node.is_leaf():
                new_parent = AABBNode()
                new_parent.parent = parent_node.parent
                new_parent.set_branch(node, parent_node)
                parent[index] = new_parent
            # if parent node is branch, compare the children's volume increase
            # and insert the node to the smaller volume increased child
            else:
                aabb0 = parent_node.children[0].aabb
                aabb1 = parent_node.children[1].aabb
                volume_diff0 = AABB.union([aabb0, node.aabb]).volume() - aabb0.volume()
                volume_diff1 = AABB.union([aabb1, node.aabb]).volume() - aabb1.volume()
                
                if volume_diff0 < volume_diff1:
                    stack.append((node, parent_node.children, 0))
                else:
                    stack.append((node, parent_node.children, 1))
                
            # update the parent nodes' AABB
            while parent_node:
                parent_node.update_aabb()
                if parent_node.parent:
                    parent_node = parent_node.parent[0]
                else: parent_node = None
                
    def remove(self, aabb):
        # remove the two-way link of the node
        # and remove the node from the tree
        self.tree_updated = True
        node = aabb.node
        node.data = None
        aabb.node = None
        self.remove_node(node)
        
    def remove_node(self, node):
        if node.parent:
            parent = node.parent[0]
            sibling = node.get_sibling()
            # if node's grandparent exists, delete the parent node
            # and attach the sibling node as the grandparent's child node
            if parent.parent:
                grandparent_node = parent.parent[0]
                sibling.parent[0] = grandparent_node
                if parent == grandparent_node.children[0]:
                    grandparent_node.children[0] = sibling
                else: grandparent_node.children[1] = sibling
                # update the grandparent nodes' AABB
                while grandparent_node:
                    grandparent_node.update_aabb()
                    if grandparent_node.parent:
                        grandparent_node = grandparent_node.parent[0]
                    else: grandparent_node = None
            # if there is no grandparent, make the sibling node as the root
            else:
                self.root[0] = sibling
                sibling.parent = []
            node.parent = []
        # if there is nothing other than the node, empty the tree
        else:
            self.root = []
                        
    def clear_collision_check(self, node):
        # change all collision checks to False
        stack = [node]
        while stack:
            node = stack.pop()
            node.collision_check = False
            if not node.is_leaf():
                for child_node in node.children:
                    stack.append(child_node)
            
    def child_collision_check(self, node, deque):
        # check node's internal collision if it is not checked
        if not node.collision_check:
            deque.insert(0,(node.children[0], node.children[1]))
            node.collision_check = True
            
    def compute_pairs(self, node0, node1):
        deque = [(node0, node1)]
        while deque:
            node0, node1 = deque.pop(0)
            if node0.is_leaf():
                # if all nodes are leaf, do AABB collision check
                if node1.is_leaf():
                    if node0.aabb.intersects(node1.aabb):
                        self.collision_pairs.append([node0.data.obj, node1.data.obj])
                # if node1 is branch, check its internal collision
                else:
                    self.child_collision_check(node1, deque)
                    deque.extend([(node0, child_node1) for child_node1 in node1.children])
            else:
                # if node0 is branch, check its internal collision
                if node1.is_leaf():
                    self.child_collision_check(node0, deque)
                    deque.extend([(child_node0, node1) for child_node0 in node0.children])
                # if all nodes are branch, check their internal collisions
                # then check their children's collisions
                else:
                    self.child_collision_check(node0, deque)
                    self.child_collision_check(node1, deque)
                    deque.extend([(child_node0, child_node1) for child_node0 in node0.children for child_node1 in node1.children])
                    
    def query_collision(self):
        # if tree is updated, compute collision pairs again
        if self.tree_updated:
            self.tree_updated = False
            self.collision_pairs = []
            if self.root[0].is_leaf(): return self.collision_pairs
            self.clear_collision_check(self.root[0])
            self.compute_pairs(self.root[0].children[0], self.root[0].children[1])
        return self.collision_pairs
    
    def update(self):
        if self.root[0]:
            if self.root[0].is_leaf():
                self.root[0].update_aabb()
            else:
                # for nodes that whose AABB no longer valid
                # remove the node and reinsert it
                self.invalid_nodes = []
                self.valid_check(self.root[0], self.invalid_nodes)
                
                for node in self.invalid_nodes:
                    self.remove_node(node)
                    node.update_aabb()
                    self.insert_node(node, self.root, index=0)
                    
                if self.invalid_nodes:
                    self.tree_update = True
                    
                self.invalid_nodes = []
                
    def valid_check(self, node, invalid_nodes):
        # check if actual AABB is out of node's AABB
        stack = [node]
        while stack:
            node = stack.pop()
            if node.is_leaf():
                if not node.aabb.contains(node.data):
                    invalid_nodes.append(node)
            else:
                stack.append(node.children[0])
                stack.append(node.children[1])