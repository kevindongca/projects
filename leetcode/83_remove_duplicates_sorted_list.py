# LeetCode 83 - Remove Duplicates from Sorted List
# https://leetcode.com/problems/remove-duplicates-from-sorted-list/
# Difficulty: Easy
# Approach: Hash set to track seen values, rewire next pointers to skip duplicates
# Time Complexity: O(n) | Space Complexity: O(n)
# Note: since list is sorted, optimal approach compares p.val to p.next.val directly (O(1) space)

from typing import Optional


class Solution:
    def deleteDuplicates(self, head: Optional['ListNode']) -> Optional['ListNode']:
        seen = {}
        q = None
        p = head
        while p != None:
            if p.val not in seen:
                seen[p.val] = 1
                q = p
                p = p.next
            else:
                q.next = p.next
                p = p.next
        return head
