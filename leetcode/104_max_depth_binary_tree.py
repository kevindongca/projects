# LeetCode 104 - Maximum Depth of Binary Tree
# https://leetcode.com/problems/maximum-depth-of-binary-tree/
# Difficulty: Easy
# Approach: Recursive DFS — 1 + max of left and right subtree depths
# Time Complexity: O(n) | Space Complexity: O(h) where h is tree height

from typing import Optional


class Solution:
    def maxDepth(self, root: Optional['TreeNode']) -> int:
        if root is None:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
