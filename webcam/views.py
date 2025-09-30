from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import RequestContext, loader
from django.http.response import StreamingHttpResponse
from ultralytics import YOLO, solutions
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
# LIVE STREAM # ―――――――――――――――――――――――――――――――――――――
def liveStream(camId):  
	cap = cv2.VideoCapture(camId)
	assert cap.isOpened(), "Error reading video file"
	
	# Video writer
	w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
	video_writer = cv2.VideoWriter("security_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

	from_email = "abc@gmail.com"  # the sender email address
	password = "---- ---- ---- ----"  # 16-digits password generated via: https://myaccount.google.com/apppasswords
	to_email = "xyz@gmail.com"  # the receiver email address

	# Initialize security alarm object
	securityalarm = solutions.SecurityAlarm(
	    show=True,  # display the output
	    model="yolo11n.pt",  # i.e. yolo11s.pt, yolo11m.pt
	    records=1,  # total detections count to send an email
	)
	
	securityalarm.authenticate(from_email, password, to_email)  # authenticate the email server

	# Process video
	while cap.isOpened():# Loop through the video frames
		success, im0 = cap.read()  # Read a frame from the video
		# if success:
		# 	results = model(im0)  # Run YOLO inference on the frame
		# 	annotated_frame = results[0].plot()  # Visualize the results on the frame
		# 	success, buffer = cv2.imencode('.jpg', annotated_frame)  # Convert the frame to JPEG format
		# 	yield (b'--frame\r\n'  # Yield the JPEG frame as a byte stream
		# 		   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
		if not success:
			break
		    
		results = securityalarm(im0)

	    # print(results)  # access the output
	
		video_writer.write(results.plot_im)  # write the processed frame.

	# Release the video capture object and close the display window
	cap.release()
	video_writer.release()
	cv2.destroyAllWindows()
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# DISPLAY CAMERA 1 # ――――――――――――――――――――――――――――――――――――――
def stream_1(request):
	return StreamingHttpResponse(liveStream(0), content_type='multipart/x-mixed-replace; boundary=frame')
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――
# DISPLAY CAMERA 2 # ――――――――――――――――――――――――――――――――――――――
def stream_2(request):
	return StreamingHttpResponse(liveStream(2), content_type='multipart/x-mixed-replace; boundary=frame')
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――




