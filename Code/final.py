import cv2, pickle
import numpy as np
import tensorflow as tf
from cnn_tf import cnn_model
import os
import sqlite3
from tensorflow.keras.models import load_model

"""
ASL Gesture Recognition System
==============================

To change the camera window size:
1. Modify the set_window_size() call near line 145, e.g.:
   set_window_size(1600, 900)  # For 1600x900 window
   set_window_size(800, 600)   # For smaller 800x600 window
   
2. Or use the default size (1200x720) which is larger than the original 640x480

To customize the gesture detection area:
1. Use the set_detection_area() function after set_window_size(), e.g.:
   set_detection_area(0.2, 0.1, 0.75, 0.7)  # Extra wide detection area
   set_detection_area(0.1, 0.05, 0.85, 0.8)  # Maximum coverage area
   
Default detection area: 60% width, starts at 35% from left (much wider than before)

Camera Options:
- The program will ask you to choose from multiple camera sources
- Supports local USB cameras (indices 0, 1, 2, etc.)
- Supports IP cameras (like phone camera apps)
- Has auto-detection feature to find available cameras

The window shows both camera feed and information panel side by side.
"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Window size configuration - you can modify these values
def set_window_size(width=1200, height=720):
	"""
	Set the window size for the camera display.
	Default: 1200x720 (larger than original 640x480)
	You can call this function with different values to change window size.
	"""
	global WINDOW_WIDTH, WINDOW_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT
	global x, y, w, h
	
	WINDOW_WIDTH = width
	WINDOW_HEIGHT = height
	CAMERA_WIDTH = WINDOW_WIDTH // 2
	CAMERA_HEIGHT = WINDOW_HEIGHT
	
	# Update hand detection region proportionally - INCREASED WIDTH
	x = int(CAMERA_WIDTH * 0.35)  # Changed from 0.5 to 0.35 (starts earlier from left)
	y = int(CAMERA_HEIGHT * 0.15) 
	w = int(CAMERA_WIDTH * 0.6)   # Changed from 0.4 to 0.6 (60% of camera width instead of 40%)
	h = int(CAMERA_HEIGHT * 0.6)

def set_detection_area(x_ratio=0.35, y_ratio=0.15, width_ratio=0.6, height_ratio=0.6):
	"""
	Customize the hand detection area within the camera frame.
	
	Parameters:
	- x_ratio: Starting X position as ratio of camera width (0.0 to 1.0)
	- y_ratio: Starting Y position as ratio of camera height (0.0 to 1.0) 
	- width_ratio: Width of detection area as ratio of camera width (0.1 to 1.0)
	- height_ratio: Height of detection area as ratio of camera height (0.1 to 1.0)
	
	Examples:
	- set_detection_area(0.2, 0.1, 0.7, 0.7)  # Very wide detection area
	- set_detection_area(0.4, 0.2, 0.5, 0.5)  # Smaller, centered area
	"""
	global x, y, w, h
	
	x = int(CAMERA_WIDTH * x_ratio)
	y = int(CAMERA_HEIGHT * y_ratio)
	w = int(CAMERA_WIDTH * width_ratio)
	h = int(CAMERA_HEIGHT * height_ratio)
	
	print(f"Detection area set to: x={x}, y={y}, width={w}, height={h}")

# Initialize default window size
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
CAMERA_WIDTH = WINDOW_WIDTH // 2
CAMERA_HEIGHT = WINDOW_HEIGHT

# Try to load the model, if it doesn't exist create a new one
try:
    model = load_model('asl_cnn_model.keras')
    print("Loaded existing model")
except:
    print("Model file not found. Creating a new model...")
    model, callbacks = cnn_model()
    print("New model created. You'll need to train it first.")
    # Save an empty model for now
    model.save('action.h5')

def get_hand_hist():
	try:
		with open("hist", "rb") as f:
			hist = pickle.load(f)
		return hist
	except FileNotFoundError:
		print("Hand histogram file not found. Creating a default histogram...")
		# Create a default histogram
		hist = np.array([[0, 0], [180, 256]], dtype=np.uint8)
		return hist

def get_image_size():
	# Return the size that matches the model's expected input
	return (64, 64)

image_x, image_y = get_image_size()

def keras_process_image(img):
	img = cv2.resize(img, (64, 64))  # Model expects 64x64 images
	# Ensure the image has 3 channels (RGB) as required by the model
	if len(img.shape) == 2:  # If grayscale, convert to RGB
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	elif len(img.shape) == 3 and img.shape[2] == 1:  # If single channel
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, 64, 64, 3))  # Model expects (batch, 64, 64, 3)
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
	try:
		conn = sqlite3.connect("gesture_db.db")
		cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
		cursor = conn.execute(cmd)
		for row in cursor:
			conn.close()
			return row[0]
		conn.close()
		return str(pred_class)  # Return the class number if not found in DB
	except:
		return str(pred_class)  # Fallback to class number

def get_pred_from_contour(contour, thresh):
	x1, y1, w1, h1 = cv2.boundingRect(contour)
	save_img = thresh[y1:y1+h1, x1:x1+w1]
	text = ""
	if w1 > h1:
		save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
	elif h1 > w1:
		save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
	pred_probab, pred_class = keras_predict(model, save_img)
	if pred_probab*100 > 70:
		text = get_pred_text_from_db(pred_class)
	return text

def get_operator(pred_text):
	try:
		pred_text = int(pred_text)
	except:
		return ""
	operator = ""
	if pred_text == 1:
		operator = "+"
	elif pred_text == 2:
		operator = "-"
	elif pred_text == 3:
		operator = "*"
	elif pred_text == 4:
		operator = "/"
	elif pred_text == 5:
		operator = "%"
	elif pred_text == 6:
		operator = "**"
	elif pred_text == 7:
		operator = ">>"
	elif pred_text == 8:
		operator = "<<"
	elif pred_text == 9:
		operator = "&"
	elif pred_text == 0:
		operator = "|"
	return operator

hist = get_hand_hist()

# Initialize window and hand detection region
set_window_size()  # Uses default 1200x720, you can pass custom values like set_window_size(1600, 900)

# Optional: Customize detection area further (uncomment and modify as needed)
# set_detection_area(0.25, 0.1, 0.7, 0.7)    # Even wider detection area
# set_detection_area(0.1, 0.05, 0.85, 0.8)   # Maximum coverage area
# set_detection_area(0.4, 0.2, 0.5, 0.5)     # Smaller, centered area

def get_img_contour_thresh(img):
	img = cv2.flip(img, 1)
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
	cv2.filter2D(dst,-1,disc,dst)
	blur = cv2.GaussianBlur(dst, (11,11), 0)
	blur = cv2.medianBlur(blur, 15)
	thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	thresh = cv2.merge((thresh,thresh,thresh))
	thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
	thresh = thresh[y:y+h, x:x+w]
	contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
	return img, contours, thresh

def calculator_mode(cam):
	flag = {"first": False, "operator": False, "second": False, "clear": False}
	count_same_frames = 0
	first, operator, second = "", "", ""
	pred_text = ""
	calc_text = ""
	info = "Enter first number"
	count_clear_frames = 0
	while True:
		img = cam.read()[1]
		img = cv2.resize(img, (CAMERA_WIDTH, CAMERA_HEIGHT))
		img, contours, thresh = get_img_contour_thresh(img)
		old_pred_text = pred_text
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				pred_text = get_pred_from_contour(contour, thresh)
				if old_pred_text == pred_text:
					count_same_frames += 1
				else:
					count_same_frames = 0

				if pred_text == "C":
					if count_same_frames > 5:
						count_same_frames = 0
						first, second, operator, pred_text, calc_text = '', '', '', '', ''
						flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
						info = "Enter first number"

				elif pred_text == "Best of Luck " and count_same_frames > 15:
					count_same_frames = 0
					if flag['clear']:
						first, second, operator, pred_text, calc_text = '', '', '', '', ''
						flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
						info = "Enter first number"
					elif second != '':
						flag['second'] = True
						info = "Clear screen"
						#Thread(target=say_text, args=(info,)).start()
						second = ''
						flag['clear'] = True
						try:
							calc_text += "= "+str(eval(calc_text))
						except:
							calc_text = "Invalid operation"
					elif first != '':
						flag['first'] = True
						info = "Enter operator"
						first = ''

				elif pred_text != "Best of Luck " and pred_text.isnumeric():
					if flag['first'] == False:
						if count_same_frames > 15:
							count_same_frames = 0
							first += pred_text
							calc_text += pred_text
					elif flag['operator'] == False:
						operator = get_operator(pred_text)
						if count_same_frames > 15:
							count_same_frames = 0
							flag['operator'] = True
							calc_text += operator
							info = "Enter second number"
							operator = ''
					elif flag['second'] == False:
						if count_same_frames > 15:
							second += pred_text
							calc_text += pred_text
							count_same_frames = 0	

		if count_clear_frames == 30:
			first, second, operator, pred_text, calc_text = '', '', '', '', ''
			flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
			info = "Enter first number"
			count_clear_frames = 0

		blackboard = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
		cv2.putText(blackboard, "Calculator Mode", (100, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
		cv2.putText(blackboard, "Predicted text- " + pred_text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
		cv2.putText(blackboard, "Operator " + operator, (30, 140), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 127))
		cv2.putText(blackboard, calc_text, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		cv2.putText(blackboard, info, (30, CAMERA_HEIGHT-50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255) )
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		res = np.hstack((img, blackboard))
		cv2.namedWindow("ASL Gesture Recognition - Calculator Mode", cv2.WINDOW_AUTOSIZE)
		cv2.imshow("ASL Gesture Recognition - Calculator Mode", res)
		cv2.namedWindow("Hand Detection Threshold", cv2.WINDOW_AUTOSIZE)
		cv2.imshow("Hand Detection Threshold", thresh)
		keypress = cv2.waitKey(1)
		if keypress == ord('q') or keypress == ord('t'):
			break

	if keypress == ord('t'):
		return 1
	else:
		return 0

def text_mode(cam):
	text = ""
	word = ""
	count_same_frame = 0
	while True:
		img = cam.read()[1]
		img = cv2.resize(img, (CAMERA_WIDTH, CAMERA_HEIGHT))
		img, contours, thresh = get_img_contour_thresh(img)
		old_text = text
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				text = get_pred_from_contour(contour, thresh)
				if old_text == text:
					count_same_frame += 1
				else:
					count_same_frame = 0

				if count_same_frame > 20:
					word = word + text
					if word.startswith('I/Me '):
						word = word.replace('I/Me ', 'I ')
					elif word.endswith('I/Me '):
						word = word.replace('I/Me ', 'me ')
					count_same_frame = 0

			elif cv2.contourArea(contour) < 1000:
				if word != '':
					pass  # Previously spoke the word here
				text = ""
				word = ""
		else:
			if word != '':
				pass  # Previously spoke the word here
			text = ""
			word = ""
		blackboard = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
		cv2.putText(blackboard, "Text Mode", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
		cv2.putText(blackboard, "Predicted text- " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
		cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		res = np.hstack((img, blackboard))
		cv2.namedWindow("ASL Gesture Recognition - Text Mode", cv2.WINDOW_AUTOSIZE)
		cv2.imshow("ASL Gesture Recognition - Text Mode", res)
		cv2.namedWindow("Hand Detection Threshold", cv2.WINDOW_AUTOSIZE)
		cv2.imshow("Hand Detection Threshold", thresh)
		keypress = cv2.waitKey(1)
		if keypress == ord('q') or keypress == ord('c'):
			break

	if keypress == ord('c'):
		return 2
	else:
		return 0

def get_camera_choice():
	"""
	Ask user to select camera source
	"""
	print("\n" + "="*50)
	print("ASL Gesture Recognition - Camera Selection")
	print("="*50)
	
	# Check if there's a saved preference
	saved_choice = load_camera_preference()
	
	print("Available camera options:")
	print("1. Default computer camera (index 0)")
	print("2. Secondary camera (index 1)")
	print("3. Third camera (index 2)")
	print("4. IP Camera (current: http://192.168.137.26:8080/video)")
	print("5. Custom IP Camera (enter your own URL)")
	print("6. Auto-detect available cameras")
	
	if saved_choice:
		print(f"\n💾 Last used: Option {saved_choice}")
		print("7. Use last saved choice")
	
	print("-"*50)
	
	while True:
		try:
			max_choice = 7 if saved_choice else 6
			choice = input(f"Enter your choice (1-{max_choice}): ").strip()
			if choice in [str(i) for i in range(1, max_choice + 1)]:
				choice_num = int(choice)
				if choice_num == 7 and saved_choice:
					return saved_choice
				else:
					# Save the new choice
					save_camera_preference(choice_num)
					return choice_num
			else:
				print(f"Please enter a valid choice (1-{max_choice})")
		except (ValueError, KeyboardInterrupt):
			print(f"Please enter a valid choice (1-{max_choice})")

def load_camera_preference():
	"""
	Load saved camera preference from file
	"""
	try:
		with open('camera_preference.txt', 'r') as f:
			return int(f.read().strip())
	except:
		return None

def save_camera_preference(choice):
	"""
	Save camera preference to file
	"""
	try:
		with open('camera_preference.txt', 'w') as f:
			f.write(str(choice))
	except:
		pass  # Fail silently if can't save

def setup_camera():
	"""
	Setup camera based on user choice
	"""
	choice = get_camera_choice()
	cam = None
	
	if choice == 1:
		print("Trying default camera (index 0)...")
		cam = cv2.VideoCapture(0)
		if cam.read()[0]:
			print("✓ Default camera connected successfully!")
			return cam
		else:
			print("✗ Default camera not available")
			cam.release()
			
	elif choice == 2:
		print("Trying secondary camera (index 1)...")
		cam = cv2.VideoCapture(1)
		if cam.read()[0]:
			print("✓ Secondary camera connected successfully!")
			return cam
		else:
			print("✗ Secondary camera not available")
			cam.release()
			
	elif choice == 3:
		print("Trying third camera (index 2)...")
		cam = cv2.VideoCapture(2)
		if cam.read()[0]:
			print("✓ Third camera connected successfully!")
			return cam
		else:
			print("✗ Third camera not available")
			cam.release()
			
	elif choice == 4:
		ip_url = "http://192.168.137.26:8080/video"
		print(f"Trying IP camera: {ip_url}")
		cam = cv2.VideoCapture(ip_url)
		if cam.read()[0]:
			print("✓ IP camera connected successfully!")
			return cam
		else:
			print("✗ IP camera not available - check URL and network connection")
			cam.release()
			
	elif choice == 5:
		ip_url = input("Enter your IP camera URL (e.g., http://192.168.1.100:8080/video): ").strip()
		if ip_url:
			print(f"Trying custom IP camera: {ip_url}")
			cam = cv2.VideoCapture(ip_url)
			if cam.read()[0]:
				print("✓ Custom IP camera connected successfully!")
				return cam
			else:
				print("✗ Custom IP camera not available - check URL and network connection")
				cam.release()
		else:
			print("✗ No URL provided")
			
	elif choice == 6:
		print("Auto-detecting available cameras...")
		for i in range(5):  # Check cameras 0-4
			print(f"Checking camera index {i}...")
			test_cam = cv2.VideoCapture(i)
			if test_cam.read()[0]:
				print(f"✓ Found working camera at index {i}")
				return test_cam
			else:
				test_cam.release()
		
		# If no local cameras found, try the default IP camera
		print("No local cameras found. Trying default IP camera...")
		ip_url = "http://192.168.137.26:8080/video"
		cam = cv2.VideoCapture(ip_url)
		if cam.read()[0]:
			print("✓ Default IP camera connected successfully!")
			return cam
		else:
			cam.release()
	
	# If we get here, the selected option failed
	print("\n❌ Selected camera option failed!")
	retry = input("Would you like to try a different camera? (y/n): ").strip().lower()
	if retry == 'y' or retry == 'yes':
		return setup_camera()  # Recursive call to try again
	else:
		return None

def recognize():
	# Setup camera with user choice
	print("Setting up camera...")
	cam = setup_camera()
	
	if cam is None:
		print("❌ Error: No camera could be initialized!")
		print("Please check your camera connections and try again.")
		return
		
	print("\n🎥 Camera ready! Starting gesture recognition...")
	print("Controls:")
	print("- Press 'Q' to quit")
	print("- Press 'T' to switch to calculator mode")
	print("- Press 'C' to switch to text mode")
	print("-" * 50)
	
	text = ""
	word = ""
	count_same_frame = 0
	keypress = 1
	while True:
		if keypress == 1:
			keypress = text_mode(cam)
		elif keypress == 2:
			keypress = calculator_mode(cam)
		else:
			break
	
	# Clean up camera when done
	cam.release()
	cv2.destroyAllWindows()
	print("\n👋 Thanks for using ASL Gesture Recognition!")

def main():
	"""
	Main function to run the ASL Gesture Recognition System
	"""
	print("🤖 Initializing ASL Gesture Recognition System...")
	
	# Initialize the model
	keras_predict(model, np.zeros((50, 50), dtype = np.uint8))
	print("✓ Model loaded successfully!")
	
	# Start the recognition system
	recognize()

if __name__ == "__main__":
	main()