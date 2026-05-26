# LeetCode 118 - Pascal's Triangle
# https://leetcode.com/problems/pascals-triangle/
# Difficulty: Easy
# Approach: pad previous row with zeros, sum adjacent pairs
# Time Complexity: O(n^2) | Space Complexity: O(n^2)

from typing import List


class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        arr = [[1]]
        for i in range(numRows - 1):
            temp = [0] + arr[-1] + [0]
            row = []
            for j in range(len(arr[-1]) + 1):
                row.append(temp[j] + temp[j + 1])
            arr.append(row)
        return arr
