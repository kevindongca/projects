# LeetCode 80 - Remove Duplicates from Sorted Array II
# https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/
# Difficulty: Medium
# Approach: Fast/slow pointers — j is write pointer, x is read pointer
# Time Complexity: O(n) | Space Complexity: O(1)
#
# j tracks the next write position. Allow each element if:
#   - j < 2: first two slots always accepted (can't have duplicates yet)
#   - x != nums[j-2]: current value differs from the one two slots back
#     (if equal to j-2 position, it would be a 3rd occurrence -> skip)
# This generalizes to "at most k duplicates" by replacing 2 with k.

from typing import List


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        j = 0
        for x in nums:
            if j < 2 or x != nums[j-2]:
                nums[j] = x
                j += 1
        return j
