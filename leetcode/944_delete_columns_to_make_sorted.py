# LeetCode 944 - Delete Columns to Make Sorted
# https://leetcode.com/problems/delete-columns-to-make-sorted/
# Difficulty: Easy
# Approach: For each column, check if any row is less than the row above it
# Time Complexity: O(n * m) | Space Complexity: O(1)
#
# Iterate over each column j, then down each row i (starting at 1).
# If strs[i][j] < strs[i-1][j], the column is unsorted -> count it and break.
# break skips remaining rows in that column since it's already disqualified.

from typing import List


class Solution:
    def minDeletionSize(self, strs: List[str]) -> int:
        n = 0
        for j in range(len(strs[0])):
            for i in range(1, len(strs)):
                if strs[i][j] < strs[i-1][j]:
                    n += 1
                    break
        return n
