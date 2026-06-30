# LeetCode 347 - Top K Frequent Elements
# https://leetcode.com/problems/top-k-frequent-elements/
# Difficulty: Medium
# Approach: Frequency count via hashmap, sort keys by count descending, take top k
# Time Complexity: O(n log n) | Space Complexity: O(n)
#
# counts maps each number to its frequency.
# Sort the keys by frequency (lambda x: counts[x]) descending, slice top k.
#
# Note: optimal solution uses a heap for O(n log k), since you only need the
# top k elements rather than a full sort of all unique values.

from typing import List


class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counts = {}
        for num in nums:
            if num not in counts.keys():
                counts[num] = 1
            else:
                counts[num] += 1
        sorted_counts = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
        return sorted_counts[:k]
