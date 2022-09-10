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
		self.boundaries = [-10, -10, 20, 20]
		self.boundary_patch.remove()
		padding = 1 

		# NOTE: boundaries = [x_min, y_min, x_max - x_min, y_max - y_min] ?
		self.axes.set_xlim(self.boundaries[0]-padding, self.boundaries[0]+self.boundaries[2]+padding)
		self.axes.set_ylim(self.boundaries[1]-padding, self.boundaries[1]+self.boundaries[3]+padding)

		patch = patches.Rectangle(self.boundaries[:2], *self.boundaries[2:4], fill=False, linewidth=1)
		self.boundary_patch = self.axes.add_patch(patch)

		# set barries
		self.barrier_patches = [
			patches.Circle((-5,0), radius=0.5),
			patches.Circle((5, 0), radius=0.5),
			patches.Rectangle((-0.55, 4.45), 1.1, 1.1),
			patches.Rectangle((-0.55, -5.55), 1.1, 1.1)
		]

		for patch in self.barrier_patches:
			patch.set(fill=True, color="#000")
			self.axes.add_patch(patch)

		# TODO: barries certs
		self.barrier_certs = [
		]

		# set goals areas
		self.goal_patches = [
			patches.Circle((5, 5), radius=0.25),
			patches.Circle((-5, 5), radius=0.25),
			patches.Circle((5, -5), radius=0.25),
			patches.Circle((-5, -5), radius=0.25),
		]

		for patch in self.goal_patches:
			patch.set(fill=False, color='#5af')
			self.axes.add_patch(patch)

	def set_velocities(self, velocities):
		"""
		velocites is a (N, 2) np.array contains (ω, v) of agents
		"""
		self._velocities = velocities

	def step(self, *args, **kwd):
		dxu = self._velocities
		for cert in self.barrier_certs:
			dxu = cert(dxu, poses)	

		super(Simulator, self).set_velocities(range(self.number_of_robots), dxu)
		super(Simulator, self).step(*args, **kwd)
