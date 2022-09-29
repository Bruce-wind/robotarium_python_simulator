
import numpy as np
from rps.utilities.controllers import create_clf_unicycle_position_controller
import simulator

#how to use simulator.py
sim = simulator.Simulator(2, show_figure=True)
controller = create_clf_unicycle_position_controller()

steps = 3000
# Define goal points
goal_points = np.array(np.mat('-2.5; -2.5'))
# goal_points = np.array(np.mat('4;4'))

poses, pose_of_hunter = sim.reset(np.array([[2],[2],[0]]))
reward = 0
for _ in range(steps):
	print(f"poses: {poses}, reward: {reward}")
	dxu = controller(poses.reshape(-1,1), goal_points)
	poses, pose_of_hunter, reward = sim.step(dxu)
