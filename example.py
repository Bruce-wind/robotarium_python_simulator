
import numpy as np
from rps.utilities.controllers import create_clf_unicycle_position_controller
import simulator
#how to use simulator.py
sim = simulator.Simulator(5, show_figure=True)
controller = create_clf_unicycle_position_controller()

steps = 3000
# Define goal points
goal_points = np.array(np.mat('-5 5 5 5 5; 5 -5 5 5 5; 0 0 0 0 0'))

for _ in range(steps):
	poses = sim.get_poses()

	dxu = controller(poses, goal_points[:2][:])
	print(dxu)
	sim.set_velocities(dxu)
	sim.step()
