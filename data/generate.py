#!/usr/bin/python

import sys
import random


y = ""
for i in range(len(x)):

    a = x[i]

    if random.random() < 0.1:
        continue

    if random.random() < 0.1:
        y += random.choice(['A', 'C', 'T', 'G'])
    else:
        y += a

print h, y.strip()
