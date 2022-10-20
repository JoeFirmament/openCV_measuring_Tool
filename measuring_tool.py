"""
Filename: Degree.py
1 识别绿色夹爪，抠绿，找出矩形四个点。
2  利用红色通道，识别布草轮廓



"""

#横排两个窗口，一个原始图像，一个测试图像

from re import X
from scipy.spatial.distance import euclidean
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import math
import cvui

class point:
  x = 0
  y = 0


# Points for Left Right Top and Bottom
plL = point()
plR = point()
plU = point()
plD = point()
 

def order_points(pts):
    # pts为轮廓坐标
	# 求最大内接矩形
    # 返回列表中存储元素分别为左上角，右上角，右下角和左下角
    rect = np.zeros((4, 2), dtype = "float32")
    # 左上角的点具有最小的和，而右下角的点具有最大的和
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算点之间的差值
    # 右上角的点具有最小的差值,
    # 左下角的点具有最大的差值
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # 返回排序坐标(依次为左上右上右下左下)
    return rect


# 求中点
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


WINDOW_NAME = 'OPENCV MEASURING TOOL by QiaoYu'

def main():
	
# 取一张图片作为例子
	use_canny = [False]
	use_hsv = [False]
	use_threshold = [False]
	img_path = "images/Sample.png"
	# Read image and preprocess
	image = cv2.imread(img_path)
	height,width,channels = image.shape
	frame = np.zeros(image.shape, np.uint8)
	#设置一个画布，容纳图像和UI元素
	canvas = np.zeros((height+400,width*3, 3), np.uint8)
	#UI元素的background color设置颜色
	canvas[height:(height+400),:] = [120,120,120]
	orig = image.copy()

	red_low_canny = [50]
	red_high_canny = [150]
	red_threshold = [30]
	green_h_min = [50]
	green_s_min = [30]
	green_v_min = [50]
	green_h_max = [100]
	green_s_max = [255]
	green_v_max = [255]
	cvui.init(WINDOW_NAME)

	while (True):
		frame = orig.copy()
		#green_only = frame.copy()
		if use_hsv[0]:

			# 转HSV格式，为抠绿准备
			hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
			# define range of green color in HSV
			# 定义绿色的取值范围
			lower_green = np.array([green_h_min,green_s_min,green_v_min],dtype=np.int32)
			upper_green = np.array([green_h_max,green_s_max,green_v_max],dtype=np.int32)
			# Threshold the HSV image to get only green colors
			mask = cv2.inRange(hsv, lower_green, upper_green)
			# Bitwise-AND mask and original image 
			# 绿色抠出，其他通道设置成黑色
			green_only = cv2.bitwise_and(image,image, mask= mask)
			green_only[:,:,0] = 0
			green_only[:,:,2] = 0
			green_only = cv2.cvtColor(green_only,cv2.COLOR_BGR2GRAY)
			#print("green_only",green_only.shape)
			green_cnts,hierarchy = cv2.findContours(green_only.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			green_cnts = [x for x in green_cnts if cv2.contourArea(x) > 2000]
			print("green cnts number : ",len(green_cnts))
			cv2.drawContours(green_only.copy(), green_cnts, -1, (0,250,0), 2)
			if len(green_cnts) == 2 :
				green_box = cv2.minAreaRect(green_cnts[1])
				green_box = cv2.boxPoints(green_box)
				green_box = np.array(green_box, dtype="int")
				green_box = perspective.order_points(green_box)
				print(green_box)
				# 抓夹的上沿两个点：
				image = cv2.circle(frame,tuple(green_box[0]),radius=2, color=(0,255,0),thickness=2)
				image = cv2.circle(frame,tuple(green_box[1]),radius=4, color=(0,255,0),thickness=2)
				image = cv2.circle(frame,tuple(green_box[2]),radius=6, color=(0,255,0),thickness=2)
				image = cv2.circle(frame,tuple(green_box[3]),radius=8, color=(0,255,0),thickness=2)
				cvui.printf(canvas,10,800,0.6,0xffffff,"ChromaKey Done")
				#cv2.imshow("chromaKey",orig)
				#cv2.waitKey(0)
			green_only = cv2.cvtColor(green_only,cv2.COLOR_GRAY2BGR)
			#mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
			green_res = np.hstack((frame,green_only))
			canvas[0:height,0:width*2] = green_res[:]
			cvui.printf(canvas,10,800,0.6,0xffffff,"                           ")

			cvui.printf(canvas,10,800,0.6,0x00ff00,"ChromaKey Processing")


			#canvas[0:height,0:width] = image[:]

		# 取出红通道，因为桌面是蓝色，夹子是绿色，所以红色通道中更容易取出白色物体。
		#取 Red 通道
			r = frame.copy()
			r[:,:,0] = 0
			r[:,:,1] = 0


			gray = cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)
			blur = cv2.GaussianBlur(gray, (7, 7), 0)
		#这两个参数设置比较重要
		#红通道阈值设置 ！！！！ 很重要
			ret, thresh = cv2.threshold(gray,red_threshold[0],255 ,cv2.THRESH_BINARY)

		#设置腐蚀卷积核系数  非常重要！！！ 用于去掉带子
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))  
			morph_open =cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel) 
			if(use_threshold[0] and (not use_canny[0])):
				red_thres_res = np.hstack((green_res,morph_open))
				canvas[0:height,0:width*3] = red_thres_res[:]
		# Find contours
		# 寻找边缘
			edged = cv2.Canny(morph_open, red_low_canny[0], red_high_canny[0])
			red_cnts = cv2.findContours(morph_open.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			red_cnts = imutils.grab_contours(red_cnts)
			red_cnts = [x for x in red_cnts if cv2.contourArea(x) > 1000]
		# 画出深红色边缘
			cv2.drawContours(frame, red_cnts, -1, (0,0,127), 2)



		#print(len(cnts))

		#---------------------------
		# 按照大小顺序，给轮廓排序取出
			if len(red_cnts) == 2 :
				cloth_cnt = sorted(red_cnts, key=cv2.contourArea, reverse=True)[0] #取出布草轮廓
				ref_object = sorted(red_cnts, key=cv2.contourArea, reverse=True)[1]  #取出对照物的轮廓
				rect_LRTB_cloth = order_points(cloth_cnt.reshape(cloth_cnt.shape[0], 2))

		#小的轮廓作为一个参考正方形。用来计算出每cm做占据的像素数
		# Reference object dimensions
		# Here for reference I have used a square
				ref_box = cv2.minAreaRect(ref_object)
				ref_box = cv2.boxPoints(ref_box)
				ref_box = np.array(ref_box, dtype="int")
				ref_box = perspective.order_points(ref_box)

				(tl, tr, br, bl) = ref_box
				dist_in_pixel = euclidean(tl, tr)
				dist_in_cm = 5
				pixel_per_cm = dist_in_pixel/dist_in_cm

				print("Pixel_Per_CM: ",pixel_per_cm)
				cvui.printf(canvas,10,820,0.6,0x00ff00,"Pixel_Per_CM is %s",pixel_per_cm)
		#----------------------

		# 取布草轮廓进行处理

				cloth_rect = cv2.minAreaRect(cloth_cnt)
				cloth_box = cv2.boxPoints(cloth_rect)
				cloth_box = np.int0(cloth_box)

				image = cv2.circle(frame,tuple(rect_LRTB_cloth[0]),radius=6, color=(0,0,0),thickness=2)
				image = cv2.circle(frame,tuple(rect_LRTB_cloth[1]),radius=6, color=(0,0,0),thickness=2)
				image = cv2.circle(frame,tuple(rect_LRTB_cloth[2]),radius=6, color=(0,0,0),thickness=2)
				image = cv2.circle(frame,tuple(rect_LRTB_cloth[3]),radius=6, color=(0,0,0),thickness=2)

				image = cv2.circle(frame,tuple(cloth_box[0]),radius=6, color=(255,255,0),thickness=2)
				image = cv2.circle(frame,tuple(cloth_box[1]),radius=6, color=(255,255,0),thickness=2)
				image = cv2.circle(frame,tuple(cloth_box[2]),radius=6, color=(255,255,0),thickness=2)
				image = cv2.circle(frame,tuple(cloth_box[3]),radius=6, color=(255,255,0),thickness=2)


		#计算布草左上角的点到夹爪上沿延长线的距离像素值
		#  布草的点排布情况如下：
		#  点0       点1
		#  ｜--------｜
		#  ｜        ｜
		#  ｜--------｜
		#  点4       点3

		#点到水平边距离
				distance_point2length = np.linalg.norm(np.cross(green_box[0]-green_box[1],green_box[1]-rect_LRTB_cloth[0])/np.linalg.norm(green_box[0]-green_box[1]))
		#点到垂直边距离
				distance_point2width = np.linalg.norm(np.cross(green_box[0]-green_box[3],green_box[3]-rect_LRTB_cloth[0])/np.linalg.norm(green_box[0]-green_box[3]))

		#计算点与线的位置，线的方向，以上图中点0为终点。返回值小于0在右边，等于0在线上。大于0在左边。


				is_above  = (green_box[1][0] - green_box[0][0])*(rect_LRTB_cloth[0][1]-green_box[0][1])-(rect_LRTB_cloth[0][0]-green_box[0][0])*(green_box[1][1]-green_box[0][1])
				is_above  = -1 * is_above
				is_left   = (green_box[3][0] - green_box[0][0])*(rect_LRTB_cloth[0][1]-green_box[0][1])-(rect_LRTB_cloth[0][0]-green_box[0][0])*(green_box[3][1]-green_box[0][1])

				print("distance from point to length:",distance_point2length/pixel_per_cm)
				print("distance from point to width :",distance_point2width/pixel_per_cm)
				print("isAbove",is_above)
				print("isLeft",is_left)
				cv2.drawContours(frame, cloth_cnt, -1, (0, 0, 0), 2)

				cloth_box = perspective.order_points(cloth_box)
				cX = np.average(cloth_box[:, 0])
				cY = np.average(cloth_box[:, 1])


				plL.x = cloth_box[0][0]
				plL.y = cloth_box[0][1]

				plR.x = cloth_box[1][0]
				plR.y = cloth_box[1][1]

				plU.x = cloth_box[2][0]
				plU.y = cloth_box[2][1]

				plD.x = cloth_box[3][0]
				plD.y = cloth_box[3][1]

				# Finding Height and width
				for (x, y) in cloth_box:
					cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
				(tl, tr, br, bl) = cloth_box

				(tltrX, tltrY) = midpoint(tl, tr)
				(blbrX, blbrY) = midpoint(bl, br)

				# compute the midpoint between the top-left and top-right points,
				# followed by the midpoint between the top-righ and bottom-right
				(tlblX, tlblY) = midpoint(tl, bl)
				(trbrX, trbrY) = midpoint(tr, br)

				# draw the midpoints on the image
				cv2.circle(frame, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
				cv2.circle(frame, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
				cv2.circle(frame, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
				cv2.circle(frame, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

				# draw lines between the midpoints
				cv2.line(frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
					(255, 0, 255), 2)
				cv2.line(frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
					(255, 0, 255), 2)

				# compute the Euclidean distance between the midpoints
				dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
				dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))


				if pixel_per_cm is None:
					pixel_per_cm = 27.6

				# compute the size of the object
				dimA = dA / pixel_per_cm
				dimB = dB / pixel_per_cm

				# draw the object sizes on the image
				cv2.putText(frame, "{:.1f}cm".format(dimA),
					(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
					0.65, (255, 255, 255), 2)
				cv2.putText(frame, "{:.1f}cm".format(dimB),
					(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
					0.65, (255, 255, 255), 2)

				# Finding Angle of the length (the larger side) and not of the width

				rp1 = point()
				rp2 = point()

				Angle = 0

				if(dA>=dB):
					rp1.x = tltrX
					rp1.y = tltrY
					rp2.x = blbrX
					rp2.y = blbrY
				else:
					rp1.x = tlblX
					rp1.y = tlblY
					rp2.x = trbrX 
					rp2.y = trbrY

				# Extending the line of which angle is to be calculated

				delX = (rp2.x - rp1.x)/(math.sqrt(((rp2.x-rp1.x) ** 2)+((rp2.y-rp1.y) ** 2))) 
				delY = (rp2.y - rp1.y)/(math.sqrt(((rp2.x-rp1.x) ** 2)+((rp2.y-rp1.y) ** 2)))

		#	cv2.line(orig, (int(rp1.x - delX*250), int(rp1.y - delY*250)), (int(rp2.x + delX*250), int(rp2.y + delY*250)),(205, 0, 0), 2)

				x,y,z = image.shape

				# The x axis, makes it easy to see the angle
				cv2.line(frame, (0 , int(y/3)), (int(x*20),int(y/3)),
					(0, 0, 0), 2)

				gradient = (rp2.y - rp1.y)*1.0/(rp2.x - rp1.x)*1.0
				Angle = math.atan(gradient)
				Angle = Angle*57.2958 # Radians to degree 

				if(Angle < 0):
					Angle = Angle + 180

				cv2.putText(frame, "{:.4f}".format(Angle) + " Degrees",
					(330, 460), cv2.FONT_HERSHEY_SIMPLEX,
					0.75, (0, 255, 255), 2)
			if(use_threshold[0] and use_canny[0]):
				red_thres_res = np.hstack((green_res,frame))
				canvas[0:height,0:width*3] = red_thres_res[:]
				# loop over the original points
				
		else :
				canvas[0:height,0:width] = orig[:]

		cvui.window(canvas, 00, 480, 640, 300, 'Green ChromaKey ',10)
		cvui.checkbox(canvas, 0, 510, 'Green ChromaKey', use_hsv)
		cvui.trackbar(canvas, 0, 530, 640, green_h_min, 0, 127)
		cvui.trackbar(canvas, 0, 570, 640, green_s_min, 0, 255)
		cvui.trackbar(canvas, 0, 610, 640, green_v_min, 0, 255)
		cvui.trackbar(canvas, 0, 650, 640, green_h_max, 0, 127)
		cvui.trackbar(canvas, 0, 690, 640, green_s_max, 0, 255)
		cvui.trackbar(canvas, 0, 730, 640, green_v_max, 0, 255)

		cvui.window(canvas, 640, 480, 640, 300, 'Red Channel Canny ',10)
		cvui.checkbox(canvas,640, 510, 'Use Red Channel Canny Edge', use_canny)
		cvui.trackbar(canvas,640, 550, 640, red_low_canny, 5, 150)
		cvui.trackbar(canvas,640, 580, 640, red_high_canny, 80, 300)


		cvui.window(canvas, 1280, 480, 640, 300, 'Red Channel Threshold',10)
		cvui.checkbox(canvas,1280, 510, 'Use Red Channel Threshold', use_threshold)
		cvui.trackbar(canvas,1280, 550, 640, red_threshold, 5, 150)


		
		cvui.update()
		# Show everything on the screen
		cv2.imshow(WINDOW_NAME, canvas)			
			
		# Check if ESC key was pressed
		if cv2.waitKey(20) == 27:
			break
if __name__ == '__main__' :
	main()



