# LeetCode 206 - Reverse Linked List
# https://leetcode.com/problems/reverse-linked-list/
# Difficulty: Easy
# Approach: Iterative — track prev/curr, reverse pointer each step
# Time Complexity: O(n) | Space Complexity: O(1)
#
# How it works each iteration:
# 1. Save curr.next before overwriting it
# 2. Point curr.next backward to prev
# 3. Advance prev and curr forward
# At the end prev is the new head (last node of original list)

from typing import Optional


class Solution:
    def reverseList(self, head: Optional['ListNode']) -> Optional['ListNode']:
        if head is None:
            return None
        curr, prev = head, None
        while curr is not None:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        return prev
