import pygame
import numpy as np

class PlayerBall (object):

	def __init__(self,pos,color,display_shape):
		self.pos = np.array(pos)
		self.r, self.color = 10, color
		self.score = 0
		self.display_width = display_shape[0]
		self.display_height = display_shape[1]

	def update_position(self, x_change, y_change):
		if (self.pos[0] + x_change < self.display_width - self.r) and (self.pos[0] + x_change > self.r):
			self.pos[0] += x_change
		if (self.pos[1] + y_change < self.display_height - self.r) and (self.pos[1] + y_change > self.r):
			self.pos[1] += y_change

	def draw(self,displayobj):
		#assert displayobj.get_width() == self.display_width
		#assert displayobj.get_height() == self.display_height
		pygame.draw.circle(displayobj, self.color, [int(self.pos[0]),int(displayobj.get_height()-self.pos[1])],self.r,0)
		return True
