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




SHOW = 1



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
	
	history_length = 16
	point_history = deque(maxlen=history_length)

	screen_size_x, screen_size_y = pyautogui.size()
	dx = screen_size_x / VIDEO_WIDTH
	dy = screen_size_y / VIDEO_HEIGHT
	
	while 1:
		success, img = cap.read()
		img = detector.find_hands(img, draw=SHOW)

		lm_list = detector.find_position(img, draw=SHOW)
		if lm_list:
			n0, n5 = lm_list[0], lm_list[5]
			last.pop(0)
			last.append(n5[1] - n0[1])
			if abs(last[0] - last[-1]) > 100:
				if last[0] < 0 and last[-1] > 0:
					print("Gesture 1 Detected")
				elif last[0] > 0 and last[-1] < 0:
					print("Gesture 2 Detected")
			
			# landmark_list = calc_landmark_list(img, lm_list)
			landmark_list = [[x, y] for _, x, y in lm_list]
			pre_processed_landmark_list = pre_process_landmark(landmark_list)

			hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
			if hand_sign_id == 2:  # Point gesture
				point_history.append(lm_list[8][1:])
			else:
				point_history.append([0, 0])

			if hand_sign_id == 2:
				# Mirror Pointer on X
				print(point_history[-1])
				pyautogui.moveTo(point_history[-1][0] * dx, point_history[-1][1] * dy)


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
