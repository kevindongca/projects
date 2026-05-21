# LeetCode 104 - Maximum Depth of Binary Tree
# https://leetcode.com/problems/maximum-depth-of-binary-tree/
# Difficulty: Easy
# Approach: Recursive DFS with explicit left/right branching
# Note: TLEs on deep trees — use the one-liner version instead
# Time Complexity: O(n) | Space Complexity: O(h)

from typing import Optional


class Solution:
    def maxDepth(self, root: Optional['TreeNode']) -> int:
        if root is None:
            return 0
        if root.left is None:
            return 1 + self.maxDepth(root.right)
        elif root.right is None:
            return 1 + self.maxDepth(root.left)
        else:
            if self.maxDepth(root.left) < self.maxDepth(root.right):
                return 1 + self.maxDepth(root.right)
            else:
                return 1 + self.maxDepth(root.left)
