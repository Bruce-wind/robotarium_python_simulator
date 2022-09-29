import numpy as np
from matplotlib import patches

from rps.robotarium import Robotarium
from rps.utilities import barrier_certificates

class Simulator(Robotarium):
	""" HITSZ ML Lab simulator """
	def __init__(self, number_of_robots=1, *args, **kwd):
		super(Simulator, self).__init__(number_of_robots=number_of_robots, *args, **kwd)
		self.init_environment()

	def init_environment(self):
		# reset boundaries
		self.boundaries = [-3.1, -3.1, 6.2, 6.2]
		if self.show_figure:
			self.boundary_patch.remove()
		padding = 1 

		if self.show_figure:
			# NOTE: boundaries = [x_min, y_min, x_max - x_min, y_max - y_min] ?
			self.axes.set_xlim(self.boundaries[0]-padding, self.boundaries[0]+self.boundaries[2]+padding)
			self.axes.set_ylim(self.boundaries[1]-padding, self.boundaries[1]+self.boundaries[3]+padding)

			patch = patches.Rectangle(self.boundaries[:2], *self.boundaries[2:4], fill=False, linewidth=2)
			self.boundary_patch = self.axes.add_patch(patch)

		# set barries
		self.barrier_centers = [(1, 1), (-1, 1), (0, -1)]
		self.radius = 0.1
		if self.show_figure:
			self.barrier_patches = [
				patches.Circle(self.barrier_centers[0], radius=self.radius),
				patches.Circle(self.barrier_centers[1], radius=self.radius),
				patches.Circle(self.barrier_centers[2], radius=self.radius),
				# patches.Rectangle((-0.25, 2.45), 0.55, 0.55),
				# patches.Rectangle((-0.25, -2.55), 0.55, 0.55)
			]

			for patch in self.barrier_patches:
				patch.set(fill=True, color="#000")
				self.axes.add_patch(patch)

			# TODO: barries certs
			self.barrier_certs = [
			]

		# set goals areas
			self.goal_patches = [
				# patches.Circle((4, 4), radius=0.24),
				# patches.Circle((-4, 4), radius=0.24),
				# patches.Circle((4, -4), radius=0.24),
				patches.Circle((-2.5, -2.5), radius=0.2),
			]

			for patch in self.goal_patches:
				patch.set(fill=False, color='#5af')
				self.axes.add_patch(patch)

	def set_velocities(self, velocities):
		"""
		velocites is a (N, 2) np.array contains (ω, v) of agents
		"""
		self._velocities = velocities

	def _step(self, *args, **kwd):
		dxu = self._velocities
		if self.show_figure:
			for cert in self.barrier_certs:
				dxu = cert(dxu, poses)	

		super(Simulator, self).set_velocities(range(self.number_of_robots), dxu)
		super(Simulator, self).step(*args, **kwd)

	def step(self, action):
		# compute hunter's action
		poses = self.get_poses()
		dxu_hunter = self.hunter_policy(poses[:, 1].reshape(-1,1), poses[:2, 0].reshape(-1,1))
		dxu = np.concatenate([action.reshape(-1,1), dxu_hunter], axis=1)

		# make a step
		self.set_velocities(dxu)
		self._step()

		# collision detect
		for robot in range(2):
			#collision with boundaries
			padding = 0.1
			self.poses[0, robot] = self.poses[0, robot] if self.poses[0, robot] > self.boundaries[0]+padding else self.boundaries[0] + padding
			self.poses[0, robot] = self.poses[0, robot] if self.poses[0, robot] < self.boundaries[0] + self.boundaries[2] - padding else self.boundaries[0] + self.boundaries[2] - padding
			self.poses[1, robot] = self.poses[1, robot] if self.poses[1, robot] > self.boundaries[1] + padding else self.boundaries[1] + padding 
			self.poses[1, robot] = self.poses[1, robot] if self.poses[1, robot] < self.boundaries[0] + self.boundaries[3] - padding else self.boundaries[1] + self.boundaries[3] - padding

			# collision with barriers
			for barrier in self.barrier_centers:
				tempA = self.poses[:2, robot] - np.array(barrier)
				dist = np.linalg.norm(tempA)

				if dist < self.radius+ padding:
					tempA = tempA / dist * (self.radius + padding)
					self.poses[:2, robot] = tempA + np.array(barrier)

		# compute the reward
		reward = self.get_reward(poses[:, 0], action)

		return self.poses[:, 0], self.poses[:, 1], reward

	def reset(self, initial_conditions):
		assert initial_conditions.shape[1] > 0, "the initial conditions must not be empty"
		assert initial_conditions.shape[1] < 3, "More than 2 robot's initial conditions receive"
		if initial_conditions.shape[1] == 1:
			self.poses = np.concatenate([initial_conditions.reshape(-1, 1), np.zeros((3,1), dtype=float)], axis=1)
		elif initial_conditions.shape[1] == 2:
			self.poses = initial_conditions

		return self.poses[:, 0], self.poses[:, 1]

	def hunter_policy(self, states, positions):
		_,N = np.shape(states)
		dxu = np.zeros((2, N))

		pos_error = positions - states[:2][:]
		rot_error = np.arctan2(pos_error[1][:],pos_error[0][:])
		dist = np.linalg.norm(pos_error, axis=0)

		dxu[0][:]=0.8*( dist + 0.2)*np.cos(rot_error-states[2][:])
		dxu[1][:]=3* dist  *np.sin(rot_error-states[2][:])

		return dxu

	############### Add Your Code Here ##############################
	def get_reward(self, state, action):
		# add you own reward function here
		hunter_state = self.poses[:, 1]
		reward = np.linalg.norm(hunter_state - state)
		return reward