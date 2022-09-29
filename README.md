[toc]

# 环境配置介绍

## 安装（windows 下)

> 其他系统下的安装可以看英文版的 README.md.

依赖库：

- numpy
- matplotlib
- scipy
- cvxopt (可选)

推荐使用 anaconda 创建一个python虚拟环境进行安装。在终端切换到创建的python环境中，运行下列命令安装依赖库 （也可以直接在 anaconda 图形化界面中直接搜索安装）。

- 使用 pip 命令安装:
    ```bash
    pip install numpy
    pip install matplotlib
    pip install scipy
    pip install cvxopt --可选
    ```
- 或者使用 conda 命令安装:
    ```bash
    conda install numpy
    conda install matplotlib
    conda install scipy
    ```

> 如果安装速度较慢可以尝试换清华源后进行下载。
> [pip 换源](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/): https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
> [conda 换源](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/): https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/

### 将仿真库安装到环境中

将该仓库克隆到本地后，在终端（命令行）切换到 python 虚拟环境，并进入该仓库目录（含有 setup.py ），运行 `pip install .`。

## 测试

在终端进到仓库目录下，运行：

```bash
python example.py
```

## Examples

小车的模型为：

$$
\begin{equation}
    \begin{split}
        x_{k+1} &= x_k + \Delta t * v_k * \cos(\theta_k)\\
        y_{k+1} &= y_k + \Delta t * v_k * \sin(\theta_k)\\
        \theta_{k+1} &= \theta_k + \Delta t * w_k \\
    \end{split}
\end{equation}
$$

在 `example.py` 文件中的代码为

```py

import numpy as np
from rps.utilities.controllers import create_clf_unicycle_position_controller
import simulator

#how to use simulator.py
sim = simulator.Simulator(2, show_figure=True)
controller = create_clf_unicycle_position_controller()

steps = 3000
# Define goal points
goal_points = np.array(np.mat('-2.5; -2.5'))

poses, pose_of_hunter = sim.reset(np.array([[2],[2],[0]]))
for _ in range(steps):
	print(f"poses: {poses}")
	dxu = controller(poses.reshape(-1,1), goal_points)
	poses, pose_of_hunter, reward = sim.step(dxu)
```

我们来逐行分析代码

```py
import numpy as np
```
导入进行矩阵运算的库

```py
from rps.utilities.controllers import create_clf_unicycle_position_controller
import simulator
```
第一行导入仿真库自带的控制器（用不上），然后导入仿真环境库。

```py
#how to use simulator.py
sim = simulator.Simulator(2, show_figure=True)
controller = create_clf_unicycle_position_controller()
```
初始化仿真环境，初始化2个小车( 一个追击者，一个猎物), 并渲染出画面（如果要强化训练的话设为 False 停止渲染加快环境运行速度）。初始化例子的控制器。

```py
steps = 3000
# Define goal points for controller
goal_points = np.array(np.mat('-2.5; -2.5'))
```
设置实验参数，运行 3000 步。控制器参数可以不用管。

```py
poses, pose_of_hunter = sim.reset(np.array([[2],[2],[0]]))
```

重置小车环境，设置逃跑者的状态，并获取小车和追击者的状态。追击者状态默认为[0,0,0]
> pose = [x, y, $\theta$]
> 
> 也可以同时设置追击者的位置，如：
> ```py
> # 追击者的状态被设置为[1, 1, 0]
> sim.reset(np.array([[2, 1], [2, 1], [0, 0]]))
> ```

```py
for _ in range(steps):
	print(f"poses: {poses}")
	dxu = controller(poses.reshape(-1,1), goal_points)
	poses, pose_of_hunter, reward = sim.step(dxu)
```
获取控制输入 `dxu`, 并输入到环境中，使环境时间向前走一步, 获得当前状态下$x_t$采取动作$u_t$的奖励$R(x_t, u_t)$, 并得到下一时刻的小车和追击者的状态.
> dxu = [v, w] 
> 奖励函数需要自己定义，可以将奖励函数写在 `simulator.py` 文件中的 `get_reward` 函数中，也可以自己在循环中添加奖励函数。

### 实验参数

| parameter | value |
| --- | --- |
| time step $\Delta t$ | 0.033 sec |
| $v_{max}$ -- prey | 0.2 m/s |
| $v_{min}$ -- prey | -0.2 m/s |
| $w_{max}$ -- prey | 3.63 rad / s|
| $w_{max}$ -- prey | -3.63 rad / s|
| $v_{max}$ -- hunter | 0.23 m/s |
| $v_{min}$ -- hunter | -0.23 m/s |
| $w_{max}$ -- hunter | 2.56 rad / s|
| $w_{max}$ -- hunter | -2.56 rad / s|
| $x_{max}$ | 3 m|
| $x_{min}$ | -3 m|
| $y_{max}$ | 3 m|
| $y_{min}$ | -3 m|
| $\theta_{max}$ | $\infin$ rad |
| $\theta_{min}$ | $-\infin$ rad|
| radius of barriers | 0.2 m |
| position of barriers | (-1, 1), (1, 1), (0, -1) unit: m|
| center of goal area | (-2.5, -2.5) unit: m|
| radius of goal area | 0.2 m |