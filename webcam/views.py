from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import RequestContext, loader
from django.http.response import StreamingHttpResponse
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import datetime
import time

# HOME PAGE # ――――――――――――――――――――――――――――――――――――――――――――
def index(request):
	template = loader.get_template('index.html')
	return HttpResponse(template.render({}, request))
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# CAMERA 1 PAGE ――――――――――――――――――――――――――――――――――――――――――
def camera_1(request):
	template = loader.get_template('camera1.html')
	return HttpResponse(template.render({}, request))
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# CAMERA 2 PAGE ――――――――――――――――――――――――――――――――――――――――――
def camera_2(request):
	template = loader.get_template('camera2.html')
	return HttpResponse(template.render({}, request))
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# YOLO MODEL ―――――――――――――――――――――――――――――――――――――――――――――
model = YOLO("yolo11n.pt")
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# LIVE STREAM # ―――――――――――――――――――――――――――――――――――――
def liveStream(camId):  
	# cam_id = 0
	cap = cv2.VideoCapture(camId)
	
	while cap.isOpened():# Loop through the video frames
		success, frame = cap.read()# Read a frame from the video
		if success:
			results = model(frame)# Run YOLO inference on the frame
			annotated_frame = results[0].plot()# Visualize the results on the frame
			# cv2.imshow("YOLO Inference", annotated_frame)# Display the annotated frame
			success, buffer = cv2.imencode('.jpg', annotated_frame)# Convert the frame to JPEG format
			yield (b'--frame\r\n'# Yield the JPEG frame as a byte stream
				   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
			# if cv2.waitKey(1) & 0xFF == ord("q"):# Break the loop if 'q' is pressed
			# 	break
		else:
			# Break the loop if the end of the video is reached
			break

	# Release the video capture object and close the display window
	cap.release()
	cv2.destroyAllWindows()
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# DISPLAY CAMERA 1 # ――――――――――――――――――――――――――――――――――――――
def stream_1(request):
	return StreamingHttpResponse(liveStream(0), content_type='multipart/x-mixed-replace; boundary=frame')
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# DISPLAY CAMERA 2 # ――――――――――――――――――――――――――――――――――――――
def stream_2(request):
	return StreamingHttpResponse(liveStream(2), content_type='multipart/x-mixed-replace; boundary=frame')
# # ――――――――――――――――――――――――――――――――――――――――――――――――――――――――
