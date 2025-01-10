import cv2
import utils
import copy
import itertools
import pyautogui
import platform
from collections import deque

from clock import Clock
from hand_model import HandDetector
from model import KeyPointClassifier


############################## Options ##############################

SHOW = 1
SMOOTHING = 3
FPS_LOCK = 30
VELOCITY_THRESHOLD = 100
COOLDOWN_GESTURE_CONST = 0.5
LM_HISTORY_LEN = 5

############################## Globals ##############################

COOLDOWN_GESTURE = 0

if platform.system() == 'Windows':
	camera_index = 0
else:
	camera_index = utils.choose_camera()
CAP = cv2.VideoCapture(camera_index)

LM_HISTORY = deque(maxlen=LM_HISTORY_LEN)
DETECTOR = HandDetector()
KEYPOINT_CLASSIFIER = KeyPointClassifier()
CLOCK = Clock(FPS_LOCK)

############################## Constants ##############################

OPEN_HAND_GESTURE = 0
POINT_GESTURE = 2

VIDEO_WIDTH = CAP.get(cv2.CAP_PROP_FRAME_WIDTH)
VIDEO_HEIGHT = CAP.get(cv2.CAP_PROP_FRAME_HEIGHT)
screen_size_x, screen_size_y = pyautogui.size()
DX = screen_size_x / VIDEO_WIDTH
DY = screen_size_y / VIDEO_HEIGHT

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


def show(img, fps):
	cv2.putText(img, str(round(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
	cv2.imshow("Image", img)
	cv2.waitKey(1)


def main():
	pTime, cTime = 0, 0
	prev_loc_x, prev_loc_y = 0, 0

	while 1:
		success, img = CAP.read()
		img = cv2.flip(img, 1)
		img = DETECTOR.find_hands(img, draw=SHOW)
		lm_list = DETECTOR.find_position(img, draw=SHOW)

		if lm_list:
			n0, n5 = lm_list[0], lm_list[5]
			LM_HISTORY.append(n5[0] - n0[0])

			pre_processed_landmark_list = pre_process_landmark(lm_list)
			hand_sign_id = KEYPOINT_CLASSIFIER(pre_processed_landmark_list)
			
			if hand_sign_id == POINT_GESTURE:
				x = prev_loc_x + (lm_list[8][0]*DX - prev_loc_x) / SMOOTHING
				y = prev_loc_y + (lm_list[8][1]*DY - prev_loc_y) / SMOOTHING
				pyautogui.moveTo(int(x), int(y), _pause=False)
				prev_loc_x = x
				prev_loc_y = y

			elif hand_sign_id == OPEN_HAND_GESTURE:
				velocity = LM_HISTORY[-1] - LM_HISTORY[0]
				if abs(velocity) > VELOCITY_THRESHOLD and len(LM_HISTORY) > 5:
					if LM_HISTORY[0] < 0 and LM_HISTORY[-1] > 0:
						gesture(left=True)
					elif LM_HISTORY[0] > 0 and LM_HISTORY[-1] < 0:
						gesture(left=False)

		fps = CLOCK.tick()

		if SHOW:
			show(img, fps)


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print("Exiting...")
