# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:31:06 2020

@author: alfre
"""

import pandas as pd
import numpy as np

class position(object):
    """
    Determine the strategy position every trading moment.
    
    Parameters
    ----------
    signal: DatetimeIndex DataFrame shape(date, 1)
        the signal of whether to buy, hold or sell every trading 
        moment. DataFrame with DatetimeIndex and the values as 1
        (long), 0(close) or -1(short).
    original_position: int (0 or 1)
        the original position of the strategy, 1 for long position
        or 0 for empty position. 1 by default.
    smooth_len: positive int
        the smooth time length that triggers the position changing
        motion. 1 by default.
    
    Attributes
    ----------
    position: DatetimeIndex DataFrame shape(date, 1)
        the position the strategy holds every trading moment. DataFrame
        with DatetimeIndex and the values as 1(long), 0(close) 
        or -1(short).
        
    """
    
    def __init__(self, signal, original_position=1, smooth_len=1):
        self.signal = signal
        self.original_position = original_position
        self.smooth_len = smooth_len
        self.position = pd.DataFrame(index=signal.index, columns=['position'])
        
    def change_position(self):
        self.position = self.position.reset_index()
        self.signal = self.signal.reset_index()
        if self.smooth_len == 1:
            self.position.loc[0, 'position'] = self.signal.loc[0, 'signal']
        else:
            self.position.loc[0: self.smooth_len-1, 'position'] = self.original_position
        for i in range(self.smooth_len-1, self.position.shape[0]):
            prev_pos = self.position.shift(1)
            prev_pos.loc[0, 'position'] = self.original_position
            if ((self.signal.loc[i-self.smooth_len+1:i, 'signal'] == self.signal.loc[i, 'signal']).all()) & (prev_pos.loc[i, 'position'] != self.signal.loc[i, 'signal']):
                self.position.loc[i, 'position'] = self.signal.loc[i, 'signal']
            else:
                self.position.loc[i, 'position'] = prev_pos.loc[i, 'position']
        self.position = self.position.set_index('date', drop=True)
        self.signal = self.signal.set_index('date', drop=True)