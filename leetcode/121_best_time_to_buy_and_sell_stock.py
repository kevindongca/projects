# LeetCode 121 - Best Time to Buy and Sell Stock
# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
# Difficulty: Easy
# Approach: Track running minimum price and best profit in one pass
# Time Complexity: O(n) | Space Complexity: O(1)
#
# At each day, the best profit if selling today = price - min_price_seen_so_far.
# Keep a running min and update best profit each step.
# No need to track max separately — profit formula handles it implicitly.

import math
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price = math.inf
        best = 0
        for price in prices:
            min_price = min(min_price, price)
            best = max(best, price - min_price)
        return best
