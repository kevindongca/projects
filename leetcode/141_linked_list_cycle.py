# LeetCode 141 - Linked List Cycle
# https://leetcode.com/problems/linked-list-cycle/
# Difficulty: Easy
# Approach: Floyd's cycle detection (tortoise and hare) — fast/slow pointers
# Time Complexity: O(n) | Space Complexity: O(1)
#
# slow moves 1 step, fast moves 2 steps each iteration.
# If a cycle exists, fast will eventually lap slow and they meet at the same node.
# If no cycle, fast hits None and we return False.
# Use `is` not `==` to compare node identity, not values.

from typing import Optional


class Solution:
    def hasCycle(self, head: Optional['ListNode']) -> bool:
        slow = head
        fast = head
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True
        return False
