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

Let's try and use this one to check the correctness implementation of the algorithms https://github.com/dennybritz/reinforcement-learning

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

### SARSA methods

There is some thing that I dont know understand, mainly why do they call an `agent.setExperience` with none values at the end, and what is the main difference with Ql

### MonteCarlo Control with epsilon soft policies

The implemented version is the very first working version, therefore not the most optimal, its performance is


| metric | value |
| --- | --- |
| TotalFrames | 156006 |
| Avg Frames Per Trial | 312.0 |
| Avg Frames per Goal  | 394.8 |
| Trials | 500 |
| Goals  | 49  |
| Defense Captured | 1 |
| Balls out of Bounds | 450 |
| Out of time | 0 |

## Exercise 3: Asyncronous Reinforcement Learning

The implemented version is the very first working version, therefore not the most optimal, its performance is

### To-Do

- [x] Use one hot encoding
- [x] Remove the gradients to the target (with `detach` or `need_grads=False`
- [x] Use `torch.Tensor([rewards])
- [x] Try different `I_async` and `I_something else`
- [x] Run without using the greedy policy and compare trained and not-trained models
- [x] Save the network
- [ ] Figure out how to know if player has the ball and add it to the state
- [ ] Suppose everything is 2D

### Metrics 

Without learning

| metric | value |
| --- | --- |
| TotalFrames | 616553 |
| Avg Frames Per Trial | 123.3 |
| Avg Frames per Goal  | 185.0 |
| Trials | 5001 |
| Goals  | 1811  |
| Defense Captured | 177 |
| Balls out of Bounds | 2994 |
| Out of time | 19 |

With learning

| metric | value |
| --- | --- |
| TotalFrames | xxx |
| Avg Frames Per Trial | xxx |
| Avg Frames per Goal  | xxx |
| Trials | xxx |
| Goals  | xxx |
| Defense Captured | xxx |
| Balls out of Bounds | xxx |
| Out of time | xxx |
