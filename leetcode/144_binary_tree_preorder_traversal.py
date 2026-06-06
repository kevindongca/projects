# LeetCode 144 - Binary Tree Preorder Traversal
# https://leetcode.com/problems/binary-tree-preorder-traversal/
# Difficulty: Easy
# Approach: Recursive DFS — root, left, right
# Time Complexity: O(n) | Space Complexity: O(h) where h is tree height
#
# Preorder: visit root first, then recurse left, then recurse right.
# [root.val] + left_result + right_result builds the list bottom-up via recursion.
# Base case: return [] for null nodes.

from typing import Optional, List


class Solution:
    def preorderTraversal(self, root: Optional['TreeNode']) -> List[int]:
        if root != None:
            return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)
        return []
