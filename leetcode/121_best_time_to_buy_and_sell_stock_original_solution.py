# LeetCode 121 - Best Time to Buy and Sell Stock
# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
# Difficulty: Easy
# Approach: Track running min and max with current profit in one pass
# Time Complexity: O(n) | Space Complexity: O(1)
#
# At each step, update min if new low found (reset max to same price).
# Update max if new high found. curr profit = max - min, track best.
# Cleaner alternative: just track min_price and compute price - min each step.

import math
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min, max = math.inf, 0
        curr = 0
        output = 0
        for i in range(len(prices)):
            if prices[i] < min:
                min, max = prices[i], prices[i]
            if prices[i] > max:
                max = prices[i]
            curr = max - min
            if output < curr:
                output = curr
        return output


# Cleaner version (same logic, fewer variables):
# def maxProfit(self, prices: List[int]) -> int:
#     min_price = math.inf
#     best = 0
#     for price in prices:
#         min_price = min(min_price, price)
#         best = max(best, price - min_price)
#     return best
