import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time


BaseOptions = mp.tasks.BaseOptions
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class HandModel():
	def __init__(self):
		self.mp_drawing = solutions.drawing_utils
		self.mp_hands = solutions.hands
		self.results = None
		self.options = HandLandmarkerOptions(
			base_options=BaseOptions(model_asset_path='model/hand_model/hand_landmarker.task'),
			running_mode=VisionRunningMode.LIVE_STREAM,
			result_callback=self.update_result
		)
		self.landmarker = HandLandmarker.create_from_options(self.options)


	def draw_landmarks_on_image(self, rgb_image, detection_result):
		hand_landmarks_list = detection_result.hand_landmarks
		annotated_image = np.copy(rgb_image)

		for idx in range(len(hand_landmarks_list)):
			hand_landmarks = hand_landmarks_list[idx]

			hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
			hand_landmarks_proto.landmark.extend([
				landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
			])
			solutions.drawing_utils.draw_landmarks(
				annotated_image,
				hand_landmarks_proto,
				solutions.hands.HAND_CONNECTIONS,
				solutions.drawing_styles.get_default_hand_landmarks_style(),
				solutions.drawing_styles.get_default_hand_connections_style(),
			)
		return annotated_image


	def update_result(self,result, output_image: mp.Image, timestamp_ms: int):
		self.results = result


	def get_landmarks(self, frame, timestamp):
		self.mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
		self.landmarker.detect_async(self.mp_image, timestamp)
		return self.results


def main():
	video = cv2.VideoCapture(0)
	model = HandModel()

	pTime = 0
	cTime = 0

	try:
		while video.isOpened(): 
			ret, frame = video.read()
			if not ret:
				print("Ignoring empty frame")
				break

			cTime = time.time()
			fps = 1/(cTime - pTime)
			pTime = cTime
			cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

			results = model.get_landmarks(frame, int(cTime*1000))
			if not (results is None):
				annotated_image = model.draw_landmarks_on_image(frame, results)
				cv2.imshow('Show', annotated_image)
			else:
				cv2.imshow('Show', frame)

			if cv2.waitKey(5) & 0xFF == ord('q'):
				print("Closing Camera Stream")
				break

	except KeyboardInterrupt:
		print("Closing Camera Stream")

	finally:
		video.release()
		cv2.destroyAllWindows()

if __name__ == "__main__":
	main()