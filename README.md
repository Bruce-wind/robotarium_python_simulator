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
sim = simulator.Simulator(5, show_figure=True)
controller = create_clf_unicycle_position_controller()

steps = 3000
# Define goal points for controller
goal_points = np.array(np.mat('-5 5 5 5 5; 5 -5 5 5 5; 0 0 0 0 0'))

for _ in range(steps):
	poses = sim.get_poses()

	dxu = controller(poses, goal_points[:2][:])
	print(dxu)
	sim.set_velocities(dxu)
	sim.step()
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
sim = simulator.Simulator(5, show_figure=True)
controller = create_clf_unicycle_position_controller()
```
初始化仿真环境，初始化5个小车并渲染出画面（如果要强化训练的话设为 False 加快训练过程）。初始化例子的控制器。

> API: simulator.Simulator(number_of_robots=1, *args, **kwd)

```py
steps = 3000
# Define goal points for controller
goal_points = np.array(np.mat('-5 5 5 5 5; 5 -5 5 5 5; 0 0 0 0 0'))
```
设置实验参数，运行 3000 步。控制器参数可以不用管。

```py
poses = sim.get_poses()
```
获取所有小车的状态。
> pose = [x, y, $\theta$]^T, 第 i 个小车的 pose 是 第 i 列。

```py
dxu = controller(poses, goal_points[:2][:])
print(dxu)
```
获取控制输入 `dxu`。
> dxu = [v, w]^T, 第 i 个小车的 dxu 是 第 i 列。

```py
sim.set_velocities(dxu)
sim.step()
```
将控制输入量设置到环境中，并是环境向前走一步。

### 实验参数

| parameter | value |
| --- | --- |
| $\Delta$ | 0.033 sec |
| $v_{max}$ | 0.2 m/s |
| $v_{min}$ | -0.2 m/s |
| $w_{max}$ | 3.63 rad / s|
| $w_{max}$ | -3.63 rad / s|
| $x_{max}$ | 10 m|
| $x_{min}$ | -10 m|
| $y_{max}$ | 10 m|
| $y_{min}$ | -10 m|