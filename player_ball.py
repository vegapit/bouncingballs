import pygame, numpy

class PlayerBall (object):

	def __init__(self, pos, color, display_shape):
		self.pos = numpy.array(pos, dtype=numpy.float32)
		self.score, self.r, self.color = 0, 10, color
		self.display_width = display_shape[0]
		self.display_height = display_shape[1]

	def update_position(self, x_change, y_change):
		if (self.pos[0] + x_change < self.display_width - self.r) and (self.pos[0] + x_change > self.r):
			self.pos[0] += x_change
		if (self.pos[1] + y_change < self.display_height - self.r) and (self.pos[1] + y_change > self.r):
			self.pos[1] += y_change

	def next_position(self, x_change, y_change):
		pos = self.pos.copy()
		if (self.pos[0] + x_change < self.display_width - self.r) and (self.pos[0] + x_change > self.r):
			pos[0] += x_change
		if (self.pos[1] + y_change < self.display_height - self.r) and (self.pos[1] + y_change > self.r):
			pos[1] += y_change
		return pos

	def draw(self, displayobj):
		pygame.draw.circle(displayobj, self.color, [int(self.pos[0]),int(displayobj.get_height()-self.pos[1])],self.r,0)
		return True

	
