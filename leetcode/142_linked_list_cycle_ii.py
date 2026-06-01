# LeetCode 142 - Linked List Cycle II
# https://leetcode.com/problems/linked-list-cycle-ii/
# Difficulty: Medium
# Approach: Floyd's cycle detection + second phase to find cycle entry point
# Time Complexity: O(n) | Space Complexity: O(1)
#
# Phase 1: fast (2 steps) and slow (1 step) meet somewhere inside the cycle.
# Phase 2: reset a runner to head, advance runner and slow 1 step at a time.
#          They meet exactly at the cycle's start node.
#
# Why phase 2 works:
# Let F = distance from head to cycle start, C = cycle length, k = steps into
# cycle where they meet. When slow enters cycle, fast has lapped it by F steps.
# Meeting point is (C - F % C) steps into cycle. After reset, runner travels F
# to reach cycle start; slow also travels F more steps to reach cycle start.
# They meet there.

from typing import Optional


class Solution:
    def detectCycle(self, head: Optional['ListNode']) -> Optional['ListNode']:
        slow = head
        fast = head
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                runner = head
                while runner is not slow:
                    runner = runner.next
                    slow = slow.next
                return runner
        return None
