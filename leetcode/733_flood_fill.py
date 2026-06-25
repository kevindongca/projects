# LeetCode 733 - Flood Fill
# https://leetcode.com/problems/flood-fill/
# Difficulty: Easy
# Approach: Recursive DFS from starting cell, recolor connected same-color cells
# Time Complexity: O(m*n) | Space Complexity: O(m*n) recursion stack
#
# Base cases: out of bounds, cell != initial_color, cell already == new color.
# The last check (image[sr][sc] == color) prevents infinite recursion when
# new color == initial color (would recurse forever without it).
# Recurse in all 4 directions after recoloring current cell.

from typing import List


class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        initial_color = image[sr][sc]
        def recfloodFill(image, initial_color, sr, sc, color):
            if sr < 0 or sr >= len(image) or sc < 0 or sc >= len(image[0]):
                return image
            if image[sr][sc] != initial_color or image[sr][sc] == color:
                return image
            image[sr][sc] = color
            recfloodFill(image, initial_color, sr+1, sc, color)
            recfloodFill(image, initial_color, sr-1, sc, color)
            recfloodFill(image, initial_color, sr, sc+1, color)
            recfloodFill(image, initial_color, sr, sc-1, color)
        recfloodFill(image, initial_color, sr, sc, color)
        return image
