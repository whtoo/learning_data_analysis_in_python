#!/usr/bin/python
# -*- coding:utf-8 -*-
from functools import wraps

def memo(func):
    cache = {}
    @wraps(func)
    def wrap(*args):
        if args not in cache:
           cache[args] = func(*args)
        return cache[args]
    return wrap

def fib(n):
    if n < 3:
        return 1
    else:
        return fib(n-1) + fib(n-2)
