# LeetCode 450 - Delete Node in a BST
# https://leetcode.com/problems/delete-node-in-a-bst/
# Difficulty: Medium
# Approach: Recursive BST traversal + inorder successor for two-child case
# Time Complexity: O(h) | Space Complexity: O(h) where h is tree height
#
# Three cases when node found:
# 1. Leaf node      → return None
# 2. One child      → return the existing child
# 3. Two children   → replace val with inorder successor (leftmost of right subtree)
#                     then delete successor from right subtree

from typing import Optional


class Solution:
    def deleteNode(self, root: Optional['TreeNode'], key: int) -> Optional['TreeNode']:
        if root is None:
            return None
        elif root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        elif root.val == key:
            if root.left is None and root.right is None:
                return None
            elif root.right is None and root.left is not None:
                return root.left
            elif root.left is None and root.right is not None:
                return root.right
            else:
                succ = root.right
                while succ.left is not None:
                    succ = succ.left
                root.val = succ.val
                root.right = self.deleteNode(root.right, succ.val)
        return root
