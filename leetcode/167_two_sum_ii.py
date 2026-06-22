# LeetCode 167 - Two Sum II - Input Array Is Sorted
# https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
# Difficulty: Medium
# Approach: Converging two pointers (works because array is sorted)
# Time Complexity: O(n) | Space Complexity: O(1)
#
# l starts at left (smallest), r starts at right (largest).
# If sum == target: found, return 1-indexed positions.
# If sum > target: right pointer moves left (need smaller value).
# If sum < target: left pointer moves right (need larger value).
# Guaranteed exactly one solution so no need for not-found case.
# Note: target - numbers[l] < numbers[r] is equivalent to numbers[l] + numbers[r] > target.

from typing import List


class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        l = 0
        r = len(numbers) - 1
        while l < r:
            if numbers[l] + numbers[r] == target:
                return [l+1, r+1]
            elif target - numbers[l] < numbers[r]:
                r -= 1
            else:
                l += 1
