# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2020/8/3 16:59
# @Author : ahrist
# @Email : 2693022425@qq.com
# @File : temp.py
# @Software: PyCharm

from Crypto.Hash import MD2
import hashlib

str1 = '73.25'
# m = hashlib.md5()
m = MD2.new()
b = str1.encode(encoding='utf-8')
for _ in range(100000000):
    m.update(b)
str_md5 = m.hexdigest()
print('MD5加密前为 ：' + str1)
print('MD5加密后为 ：' + str_md5)
