# Position module

This module contains the class to receive signals and give out updated positions.

Determine the strategy position every trading moment.

## Parameters

- **signal**: DatetimeIndex DataFrame shape(date, 1)
  the signal of whether to buy, hold or sell every trading 
  moment. DataFrame with DatetimeIndex and the values as 1
  (long), 0(close) or -1(short).
- **original_position**: int (0 or 1)
  the original position of the strategy, 1 for long position
  or 0 for empty position. 1 by default.
- **smooth_len**: positive int
  the smooth time length that triggers the position changing
  motion. 1 by default.

## Attributes

- **position**: DatetimeIndex DataFrame shape(date, 1)
  the position the strategy holds every trading moment. DataFrame
  with DatetimeIndex and the values as 1(long), 0(close) 
  or -1(short).