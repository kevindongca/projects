# LeetCode 100 - Same Tree
# https://leetcode.com/problems/same-tree/
# Difficulty: Easy
# Approach: Recursive DFS
# Time Complexity: O(n) | Space Complexity: O(h) where h is tree height

from typing import Optional


class Solution:
    def isSameTree(self, p: Optional['TreeNode'], q: Optional['TreeNode']) -> bool:
        # both None — reached end of both trees simultaneously, they match
        if p is None and q is None:
            return True
        # one is None but not the other — structure differs
        if p is None or q is None:
            return False
        # values differ
        if p.val != q.val:
            return False
        # recursively check left and right subtrees
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
