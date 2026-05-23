# LeetCode 110 - Balanced Binary Tree
# https://leetcode.com/problems/balanced-binary-tree/
# Difficulty: Easy
# Approach: DFS height check with -1 sentinel for early exit
# Time Complexity: O(n) | Space Complexity: O(h)
# Key insight: return -1 to signal unbalanced subtree and short-circuit
# instead of recomputing heights (which would be O(n log n))

from typing import Optional


class Solution:
    def isBalanced(self, root: Optional['TreeNode']) -> bool:

        def height(node):
            if not node:
                return 0
            left_h = height(node.left)
            if left_h == -1:
                return -1          # already unbalanced, short-circuit
            right_h = height(node.right)
            if right_h == -1:
                return -1
            if abs(left_h - right_h) > 1:
                return -1          # this node is unbalanced
            return 1 + max(left_h, right_h)

        return height(root) != -1
