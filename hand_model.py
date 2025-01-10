import mediapipe as mp
import cv2
import time
import utils

class HandDetector():
	def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_con=0.5, trac_con=0.5):
		self.mode = mode
		self.max_hands = max_hands
		self.model_complexity = model_complexity
		self.detection_con = detection_con
		self.trac_con = trac_con
		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands(mode, max_hands, model_complexity, detection_con, trac_con)
		self.mpDraw = mp.solutions.drawing_utils


	def find_hands(self, img, draw=True):
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		results = self.hands.process(imgRGB)
		self.lms = results.multi_hand_landmarks
		if self.lms:
			for handLms in self.lms:
				if draw:
					self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
		return img

	def find_position(self, img, handNo=0, draw=True):
		self.lm_list = []
		if self.lms:
			myHand = self.lms[handNo]
			for id, lm in enumerate(myHand.landmark):
				h, w, c = img.shape
				cx, cy = int(lm.x * w), int(lm.y * h)
				self.lm_list.append([cx, cy])
				if draw:
					cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)
		return self.lm_list

	

def main():
	camera_index = utils.choose_camera()
	cap = cv2.VideoCapture(camera_index)
	VIDEO_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	VIDEO_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

	pTime = 0
	cTime = 0
	detector = HandDetector(model_complexity=1)

	while 1:
		success, img = cap.read()
		img = detector.find_hands(img)
		lm_list = detector.find_position(img)

		cTime = time.time()
		fps = 1/(cTime - pTime)
		pTime = cTime

		# print(fps)
		cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
		cv2.imshow("Image", img)
		cv2.waitKey(1)


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		pass
