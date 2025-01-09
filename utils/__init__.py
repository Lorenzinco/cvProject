import platform

system = platform.system()
if system == 'Windows':
	choose_camera = lambda: 0
else:
	from utils.choose_camera import choose_camera
