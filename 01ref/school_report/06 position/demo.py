# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:06:37 2020

@author: alfre
"""

import pandas as pd
from position import position

a = pd.DataFrame(index=range(5), columns=['date', 'signal'])
a.date = range(5)
a.signal = [0, 1, -1, 0, 1]
a = a.set_index('date', drop=True)

pos = position(a, smooth_len=2)
pos.change_position()