# LeetCode 136 - Single Number
# https://leetcode.com/problems/single-number/
# Difficulty: Easy
# Approach: Bit manipulation — XOR all elements
# Time Complexity: O(n) | Space Complexity: O(1)
#
# XOR properties:
#   a ^ a = 0  (same number cancels itself out)
#   a ^ 0 = a  (zero is the identity)
#   XOR is commutative and associative (order doesn't matter)
#
# So XOR-ing all elements causes every duplicate pair to cancel to 0,
# leaving only the single number. e.g. [2,2,1] -> 0^2^2^1 = 0^0^1 = 1

from typing import List


class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        result = 0
        for n in nums:
            result ^= n
        return result
