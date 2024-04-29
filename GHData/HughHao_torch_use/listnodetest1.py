# -*- coding: utf-8 -*-
# @Time : 2022/4/12 16:17
# @Author : hhq
# @File : listnodetest1.py
from listnode import Node, SingleLinkList
a1 = Node(1)
a2 = Node(2)
a3 = Node(3)
a4 = Node(4)

print(a1.elem)  # 1
print(a1.next)  # None
a1.next = a2  # 2
print(a1.next.elem)  # 2
print(a2.elem)  # 2
a2.next = a3
a3.next = a4
# print(SingleLinkList)
lin1 = SingleLinkList(a2)  # 添加和删除操作将从ak开始
print(lin1.length())  # 从a2开始的链表长度 3
# lin1.is_empty()
print(lin1.travel())  # 2 3 4
print(a3.next.elem)  # 4
print(a2.next.next.elem)  # 4
# print()
lin1.add(4)
lin1.append(6)
lin1.remove(2)
# print(lin1.travel())
