# Labeling

Labels are event outcomes, not next-candle direction.

For each 5m event at time t:

1. Define an entry time t + execution delay.
2. Look forward through the 1m path only until the time barrier or session close.
3. Evaluate direction-specific target/stop barriers.
4. Record the first clean barrier event.
5. Mark same-minute target/stop collisions as ambiguous; they are excluded from the
   default training set rather than being assigned an arbitrary side.
6. Store MFE/MAE and barrier timing independently from the categorical label.

Baseline distances:

- target: 200 points
- stop: 70 points
- horizon: 75 x 5m bars
- entry delay: 1 minute

Target ladders are supported so target/stop selection can be treated as an
experiment instead of an undocumented constant.
