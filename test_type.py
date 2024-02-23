#!/usr/bin/env python3
import cv2

src = cv2.imread("/home/hi/문서/Self_Driving_Car-master/CarND-Advanced-Lane-Lines/examples/color_fit_lines.jpg", cv2.IMREAD_COLOR)
dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

print(src[1,2,0])

print(type(src))
print(type(dst))
