import cv2
from simple_term_menu import TerminalMenu

def returnCameraIndexes():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

def choose_camera():
	indexes = returnCameraIndexes()
	menu = TerminalMenu(["Camera " + str(i) for i in indexes])
	menu_entry_index = menu.show()
	return indexes[menu_entry_index]