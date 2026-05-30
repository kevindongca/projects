# LeetCode 226 - Invert Binary Tree
# https://leetcode.com/problems/invert-binary-tree/
# Difficulty: Easy
# Approach: Recursive DFS — swap children at every node
# Time Complexity: O(n) | Space Complexity: O(h)
#
# Python evaluates the right side fully before assignment
# so the simultaneous swap works correctly in one line

from typing import Optional


class Solution:
    def invertTree(self, root: Optional['TreeNode']) -> Optional['TreeNode']:
        if root is None:
            return None
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root
