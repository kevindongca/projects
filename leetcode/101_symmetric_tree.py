# LeetCode 101 - Symmetric Tree
# https://leetcode.com/problems/symmetric-tree/
# Difficulty: Easy
# Approach: Recursive DFS with mirror checker helper
# Time Complexity: O(n) | Space Complexity: O(h) where h is tree height

from typing import Optional


class Solution:
    def isSymmetric(self, root: Optional['TreeNode']) -> bool:

        def checker(root1, root2):
            # both None — symmetric at this position
            if not root1 and not root2:
                return True
            # one is None but not the other — not symmetric
            if not root1 or not root2:
                return False
            # values differ — not symmetric
            if root1.val != root2.val:
                return False
            # check outer pair and inner pair recursively
            return checker(root1.left, root2.right) and checker(root2.right, root1.left)

        return checker(root, root)
