# LeetCode 492 - Construct the Rectangle
# https://leetcode.com/problems/construct-the-rectangle/
# Difficulty: Easy
# Approach: Iterate divisors up to sqrt(area), find pair with minimum L-W difference
# Time Complexity: O(sqrt(n)) | Space Complexity: O(1)
#
# Only need to check i up to sqrt(area) since divisors come in pairs (i, area//i).
# For each valid divisor i (area % i == 0), compute how close the pair is to a square.
# Track the pair with minimum difference; min(i, area//i) = width, max = length.

import math


class Solution:
    def constructRectangle(self, area: int) -> List[int]:
        min_diff = math.inf
        width = 0
        length = 0
        for i in range(1, int(math.sqrt(area)) + 1):
            if area % i == 0 and min_diff > abs(i - (area // i)):
                min_diff = abs(i - (area // i))
                width = min(i, area // i)
                length = max(i, area // i)
        return [length, width]
