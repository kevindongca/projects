# LeetCode 112 - Path Sum
# https://leetcode.com/problems/path-sum/
# Difficulty: Easy
# Approach: DFS tracking running sum, check at leaf nodes
# Time Complexity: O(n) | Space Complexity: O(h)

from typing import Optional


class Solution:
    def hasPathSum(self, root: Optional['TreeNode'], targetSum: int) -> bool:
        total = 0

        def dfs(node, total):
            if not node:
                return False
            total += node.val
            if total == targetSum and node.left is None and node.right is None:
                return True
            if node.left is None and node.right is None:
                return False
            return dfs(node.left, total) or dfs(node.right, total)

        return dfs(root, total)
