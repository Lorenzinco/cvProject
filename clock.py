import time

ERROR = 0.001

class Clock:
	def __init__(self, fps_max):
		self.pTime = 0
		self.cTime = 0
		self.fps_max = fps_max
		self.dt_max = 1 / self.fps_max - ERROR
		self.now = time.time

	def tick(self):
		self.cTime = self.now()
		DT = self.cTime - self.pTime
		if DT < self.dt_max:
			time.sleep(self.dt_max - DT)
			self.cTime = self.now()
			DT = self.cTime - self.pTime
		self.pTime = self.cTime
		fps = 1/(DT)
		return fps
