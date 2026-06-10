# LeetCode 219 - Contains Duplicate II
# https://leetcode.com/problems/contains-duplicate-ii/
# Difficulty: Easy
# Approach: Hashmap storing last seen index of each value
# Time Complexity: O(n) | Space Complexity: O(n)
#
# last_seen maps each value to its most recently seen index.
# For each element, if it's already in last_seen AND the distance <= k, return True.
# Always update last_seen[num] to the current index (keeps it as the closest prior occurrence).

from typing import List


class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        last_seen = {}
        for i, num in enumerate(nums):
            if num in last_seen and i - last_seen[num] <= k:
                return True
            last_seen[num] = i
        return False
