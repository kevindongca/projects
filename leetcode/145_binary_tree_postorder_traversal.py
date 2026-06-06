# LeetCode 145 - Binary Tree Postorder Traversal
# https://leetcode.com/problems/binary-tree-postorder-traversal/
# Difficulty: Easy
# Approach: Recursive DFS — left, right, root
# Time Complexity: O(n) | Space Complexity: O(h) where h is tree height
#
# Postorder: recurse left, recurse right, visit root last.
# left_result + right_result + [root.val] builds the list bottom-up.
# Opposite of preorder — root goes at the end instead of the front.
# Base case: return [] for null nodes.

from typing import Optional, List


class Solution:
    def postorderTraversal(self, root: Optional['TreeNode']) -> List[int]:
        if root is not None:
            return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]
        return []
