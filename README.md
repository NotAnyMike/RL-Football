# RL for Football RoboCup 2D Half Field Offense

## Exercise 1: Simple policy iteration exercise

Based on policy iteration, which is can be expresed in solving for the optimal policy using

![eq1](https://latex.codecogs.com/gif.latex?V%28s%29%20%5Cleftarrow%20%5Cmax%20%5Csum_%7Bs%27%2Cr%7Dp%28s%27%2Cr%7Cs%2C%5Cpi%28s%29%29%5Cleft%5Br&plus;%5Cgamma%20V%28s%27%29%20%5Cright%5D)
<!---- V(s) \left \max \sum_{s',r}p(s',r|s,\pi(s))\left[r+\gamma V(s') \right] --->
until 

![eq2](https://latex.codecogs.com/gif.latex?%24%5CDelta%20%5Cleftarrow%20%5Cmax%28%5CDelta%2C%20%7Cv-V%28s%29%7C%29%24)

is small enought, in order solve a toy example, 

The optimal policy and optimal values are

![sol](Exercise1/img/exc1.png)

## Exercise 2: Different TD algorithms

### Q-learning algorithms

The implemented version is the very first working version, therefore not the most optimal, its performance is

| metric | value |
| --- | --- |
| TotalFrames | 114160 |
| Avg Frames Per Trial | 228.3 |
| Avg Frames per Goal  | 270.4 |
| Trials | 500 |
| Goals  | 47  |
| Defense Captured | 0 |
| Balls out of Bounds | 454 |
| Out of time | 0 |
