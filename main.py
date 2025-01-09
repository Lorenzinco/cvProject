import cv2
import time
import utils
from hand_model import HandDetector


SHOW = 1


def main():
	camera_index = utils.choose_camera()
	cap = cv2.VideoCapture(camera_index)

	pTime = 0
	cTime = 0
	detector = HandDetector(model_complexity=1)

	last = [0] * 5

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
