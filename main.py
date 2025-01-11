import cv2
import utils
import copy
import itertools
import pyautogui
import platform
import toml
from collections import deque
from numpy import interp

from clock import Clock
from model import HandModel
from model import KeyPointClassifier


############################## Options ##############################

config = toml.load("./config.toml")
SHOW = config.get('Settings', {}).get('show', True)
SMOOTHING = config.get('Settings', {}).get('smoothing', 3)
FPS_LOCK = config.get('Settings', {}).get('fps_lock', 30)
VELOCITY_THRESHOLD = config.get('Gestures', {}).get('velocity_threshold', 100)
COOLDOWN_GESTURE_CONST = config.get('Gestures', {}).get('cooldown_gesture_const', 0.5)
LM_HISTORY_LEN = config.get('Gestures', {}).get('lm_history_len', 5)
POINTER_SENSITIVITY = config.get('Pointer', {}).get('pointer_sensitivity', 800)
DEAD_ZONE=config.get('Pointer', {}).get('dead_zone', 0.05)

############################## Globals ##############################

COOLDOWN_GESTURE = 0

if platform.system() == 'Windows':
	camera_index = 0
else:
	camera_index = utils.choose_camera()
CAP = cv2.VideoCapture(camera_index)

LM_HISTORY = deque(maxlen=LM_HISTORY_LEN)
WLM_HISTORY = deque(maxlen=LM_HISTORY_LEN)
MODEL = HandModel()
KEYPOINT_CLASSIFIER = KeyPointClassifier()
CLOCK = Clock(FPS_LOCK)

############################## Constants ##############################

OPEN_HAND_GESTURE = 0
POINT_GESTURE = 2

VIDEO_WIDTH = CAP.get(cv2.CAP_PROP_FRAME_WIDTH)
VIDEO_HEIGHT = CAP.get(cv2.CAP_PROP_FRAME_HEIGHT)
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
DX = SCREEN_WIDTH / VIDEO_WIDTH
DY = SCREEN_HEIGHT / VIDEO_HEIGHT

############################## Functions ##############################

def pre_process_landmark(landmark_list):
	temp_landmark_list = copy.deepcopy(landmark_list)

	# Convert to relative coordinates
	base_x, base_y = 0, 0
	for index, landmark_point in enumerate(temp_landmark_list):
		if index == 0:
			base_x, base_y = landmark_point[0], landmark_point[1]

		temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
		temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

	# Convert to a one-dimensional list
	temp_landmark_list = list(
		itertools.chain.from_iterable(temp_landmark_list))

	# Normalization
	max_value = max(list(map(abs, temp_landmark_list)))

	def normalize_(n):
		return n / max_value

	temp_landmark_list = list(map(normalize_, temp_landmark_list))
	return temp_landmark_list


def gesture(left=False):
	global COOLDOWN_GESTURE
	now = CLOCK.now()
	if COOLDOWN_GESTURE + COOLDOWN_GESTURE_CONST > now:
		return
	COOLDOWN_GESTURE = now
	if left:
		direction = 'left'
	else:
		direction = 'right'
	print(f"Gesture {direction}")
	pyautogui.press(direction, _pause=False)

def get_pointing_direction(wlm_list, center_coordinates):
	wn5, wn8 = wlm_list[5], wlm_list[8]
	direction = (wn8[0] - wn5[0], wn8[1] - wn5[1], wn8[2] - wn5[2])
	norm = (direction[0]**2 + direction[1]**2 + direction[2]**2)**0.5
	if norm < DEAD_ZONE:
		# print("Dead Zone")
		return center_coordinates
	
	direction = (direction[0]/norm, direction[1]/norm, direction[2]/norm)
	new_x = center_coordinates[0] + direction[0] * POINTER_SENSITIVITY
	new_y = center_coordinates[1] + direction[1] * POINTER_SENSITIVITY
	# print(new_x,new_y, end='\r')
	return new_x, new_y

def main():
	prev_loc_x, prev_loc_y = 0, 0

	while 1:
		success, img = CAP.read()
		if not success:
			print("Ignoring empty frame")
			break

		img = cv2.flip(img, 1)
		results = MODEL.get_landmarks(img, int(CLOCK.now()*1000))

		lm_list = []
		wlm_list = []
		if not (results is None):
			if results.handedness:
				for lm in results.hand_landmarks[0]:
					lm_list.append([
						round(lm.x * VIDEO_WIDTH),
						round(lm.y * VIDEO_HEIGHT),
					])
			
				for lm in results.hand_world_landmarks[0]:
					wlm_list.append([
						round(lm.x,3),
						round(lm.y,3),
						round(lm.z,3),
					])
				# print(results.handedness[0][0].category_name)
				# print(results.hand_landmarks[0][8].x * VIDEO_WIDTH, results.hand_landmarks[0][8].y * VIDEO_HEIGHT, results.hand_landmarks[0][8].z * VIDEO_WIDTH, end='\r')
				# print(results.hand_landmarks)
				# print(results.hand_world_landmarks)

		if lm_list and wlm_list:
			n0, n5 = lm_list[0], lm_list[5]
			LM_HISTORY.append(n5[0] - n0[0])
			wn5, wn8 = wlm_list[5], wlm_list[8]
			# WLM_HISTORY.append(wn8 - wn5)

			pre_processed_landmark_list = pre_process_landmark(lm_list)
			hand_sign_id = KEYPOINT_CLASSIFIER(pre_processed_landmark_list)
			
			if hand_sign_id == POINT_GESTURE:
				# calculate the pointing direction starting from the index finger base to the tip
				center_coordinates = (n5[0], n5[1])
				x,y = get_pointing_direction(wlm_list,center_coordinates=center_coordinates)
				# x = interp(lm_list[8][0], (0, VIDEO_WIDTH), (-POINTER_SENSITIVITY, SCREEN_WIDTH+POINTER_SENSITIVITY))
				# y = interp(lm_list[8][1], (0, VIDEO_HEIGHT), (-POINTER_SENSITIVITY, SCREEN_HEIGHT+POINTER_SENSITIVITY))
				if x < 0: x = 5
				if y < 0: y = 5
				if x >= SCREEN_WIDTH: x = SCREEN_WIDTH-5
				if y >= SCREEN_HEIGHT: y = SCREEN_HEIGHT-5
				x = prev_loc_x + (x - prev_loc_x) / SMOOTHING
				y = prev_loc_y + (y - prev_loc_y) / SMOOTHING
				pyautogui.moveTo(int(x), int(y), _pause=False)
				prev_loc_x = x
				prev_loc_y = y

			elif hand_sign_id == OPEN_HAND_GESTURE:
				velocity = LM_HISTORY[-1] - LM_HISTORY[0]
				if abs(velocity) > VELOCITY_THRESHOLD and len(LM_HISTORY) == LM_HISTORY_LEN:
					if LM_HISTORY[0] < 0 and LM_HISTORY[-1] > 0:
						gesture(left=True)
					elif LM_HISTORY[0] > 0 and LM_HISTORY[-1] < 0:
						gesture(left=False)

		
		fps = CLOCK.tick()

		
		if not SHOW:
			continue

		cv2.putText(img, str(round(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
		if not (results is None):
			annotated_image = MODEL.draw_landmarks_on_image(img, results)
			cv2.imshow('Show', annotated_image)
		else:
			cv2.imshow('Show', img)

		if cv2.waitKey(5) & 0xFF == ord('q'):
			print("Closing Camera Stream")
			break


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print("Exiting...")
