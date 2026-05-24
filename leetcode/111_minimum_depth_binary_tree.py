# LeetCode 111 - Minimum Depth of Binary Tree
# https://leetcode.com/problems/minimum-depth-of-binary-tree/
# Difficulty: Easy

from typing import Optional
from collections import deque


class SolutionDFS:
    def minDepth(self, root: Optional['TreeNode']) -> int:
        if not root:
            return 0
        left  = self.minDepth(root.left)
        right = self.minDepth(root.right)
        if not root.left:  return 1 + right
        if not root.right: return 1 + left
        return 1 + min(left, right)


class SolutionBFS:
    def minDepth(self, root: Optional['TreeNode']) -> int:
        if not root:
            return 0
        queue = deque([(root, 1)])
        while queue:
            node, depth = queue.popleft()
            if not node.left and not node.right:
                return depth
            if node.left:  queue.append((node.left,  depth + 1))
            if node.right: queue.append((node.right, depth + 1))
