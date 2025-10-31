import pygame, numpy

class Ball (object):

	@staticmethod
	def check_hit(pos1, r1, pos2, r2):
		if numpy.sqrt(numpy.sum((pos1 - pos2)**2.0)) <= r1 + r2:
			return True
		else:
			return False

	def __init__(self, pos, v0, color, display_shape):
		self.pos = numpy.array(pos, dtype=numpy.float32)
		self.v = numpy.array(v0, dtype=numpy.float32)
		self.live, self.r, self.color = True, 8, color
		self.display_width = display_shape[0]
		self.display_height = display_shape[1]

	def _check_bounce(self):
		if (self.pos[1] == self.r) or (self.pos[1] == self.display_height - self.r):
			self.v[1] = -self.v[1]
		if (self.pos[0] == self.r) or (self.pos[0] == self.display_width - self.r):
			self.v[0] = -self.v[0]

	def update_position(self, dt):
		self.pos = self.pos + self.v * dt
		if self.pos[0] < self.r:
			self.pos[0] = self.r
		elif self.pos[0] > self.display_width - self.r:
			self.pos[0] = self.display_width - self.r
		if self.pos[1] < self.r:
			self.pos[1] = self.r
		elif self.pos[1] > self.display_height - self.r:
			self.pos[1] = self.display_height - self.r
		self._check_bounce()
		return self.pos

	def next_position(self, dt):
		pos = self.pos.copy()
		pos = pos + self.v * dt
		if pos[0] < self.r:
			pos[0] = self.r
		elif pos[0] > self.display_width - self.r:
			pos[0] = self.display_width - self.r
		if pos[1] < self.r:
			pos[1] = self.r
		elif pos[1] > self.display_height - self.r:
			pos[1] = self.display_height - self.r
		return pos

	def draw(self, displayobj):
		pygame.draw.circle(displayobj, self.color, [int(self.pos[0]),int(displayobj.get_height()-self.pos[1])],self.r,0)
		return True
