import cv2
import time
import utils
import copy
import itertools
import pyautogui
import platform
from collections import deque

from hand_model import HandDetector

from model import KeyPointClassifier
from model import PointHistoryClassifier
import threading




SHOW = 1
POINT_GESTURE = 2
OPEN_HAND_GESTURE = 0
VELOCITY_THRESHOLD = 235
GESTURE_DELAY = 1

gesture_found = False

def reset_gesture_found():
	global gesture_found
	time.sleep(GESTURE_DELAY)
	gesture_found = False

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
	# if max_value == 0:
	# 	return [0] * len(temp_landmark_list)

	def normalize_(n):
		return n / max_value

	temp_landmark_list = list(map(normalize_, temp_landmark_list))

	return temp_landmark_list


def calc_landmark_list(image, landmarks):
	image_width, image_height = image.shape[1], image.shape[0]

	landmark_point = []

	# Keypoint
	for _, landmark in enumerate(landmarks):
		landmark_x = min(int(landmark[1] * image_width), image_width - 1)
		landmark_y = min(int(landmark[2] * image_height), image_height - 1)
		# landmark_z = landmark.z

		landmark_point.append([landmark_x, landmark_y])

	return landmark_point


def main():
	if platform.system() == 'Windows':
		camera_index = 0
	else:
		camera_index = utils.choose_camera()
	cap = cv2.VideoCapture(camera_index)

	pTime = 0
	cTime = 0
	detector = HandDetector(model_complexity=1)

	VIDEO_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	VIDEO_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

	last = [0] * 5

	keypoint_classifier = KeyPointClassifier()
	point_history_classifier = PointHistoryClassifier()
	
	point_history_length = 16
	gesture_history_length = 5
	point_history = deque(maxlen=point_history_length)
	position_history = deque(maxlen=gesture_history_length)

	screen_size_x, screen_size_y = pyautogui.size()
	dx = screen_size_x / VIDEO_WIDTH
	dy = screen_size_y / VIDEO_HEIGHT

	global gesture_found
	
	while 1:
		
		success, img = cap.read()
		img = cv2.flip(img, 1) # mirror the image
		img = detector.find_hands(img, draw=SHOW)

		lm_list = detector.find_position(img, draw=SHOW)
		position_history.append(lm_list)
		if lm_list:
			n0, n5 = lm_list[0], lm_list[5]
			last.pop(0)
			last.append(n5[1] - n0[1])


			landmark_list = [[x, y] for _, x, y in lm_list]
			pre_processed_landmark_list = pre_process_landmark(landmark_list)

			hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
			if hand_sign_id == POINT_GESTURE:
				point_history.append(lm_list[8][1:])

			elif hand_sign_id == OPEN_HAND_GESTURE:
				point_history.append([0, 0])
				if len(position_history) == gesture_history_length:
					first_point = position_history[0]
					last_point = position_history[-1]

					# print(first_point, last_point)
					if len(first_point)>5 and len(last_point)>5:
						first_n0, last_n0 = first_point[0], last_point[0]
						first_n5, last_n5 = first_point[5], last_point[5]
						
						velocity = last_n5[1] - first_n5[1]

						if abs(velocity) > VELOCITY_THRESHOLD and not gesture_found:
							if last[0] < 0 and last[-1] > 0:
								gesture_found = True
								

								threading.Thread(target=reset_gesture_found).start()
								print("Gesture 1 Detected")
								print(velocity)
								pyautogui.press('left',_pause=False)

							elif last[0] > 0 and last[-1] < 0:
								gesture_found = True
								threading.Thread(target=reset_gesture_found).start()
								print(velocity)
								print("Gesture 2 Detected")
								pyautogui.press('right',_pause=False)
			else:
				point_history.append([0, 0])

			
			

			

			#print(hand_sign_id)

			if hand_sign_id == 2:
				# Mirror Pointer on X
				#print(point_history[-1])
				pyautogui.moveTo(point_history[-1][0] * dx, point_history[-1][1] * dy, _pause=False)


		if not SHOW:
			continue
		
		cTime = time.time()
		fps = 1/(cTime - pTime)
		pTime = cTime

		cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
		cv2.imshow("Image", img)
		cv2.waitKey(1)



if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		pass
