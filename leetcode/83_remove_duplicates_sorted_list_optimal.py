# LeetCode 83 - Remove Duplicates from Sorted List
# https://leetcode.com/problems/remove-duplicates-from-sorted-list/
# Difficulty: Easy
# Approach: Single pointer, rewire next pointer to skip duplicates
# Time Complexity: O(n) | Space Complexity: O(1)
# Since list is sorted, duplicates are always adjacent — no hash set needed

from typing import Optional


class Solution:
    def deleteDuplicates(self, head: Optional['ListNode']) -> Optional['ListNode']:
        if not head:
            return head
        p = head
        while p.next != None:
            if p.val == p.next.val:
                p.next = p.next.next  # skip duplicate, stay at p
            else:
                p = p.next            # no duplicate, move forward
        return head
