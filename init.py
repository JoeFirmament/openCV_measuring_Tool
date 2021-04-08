"""
Filename: init.py
识别图片中物体，标记各种特征点、最大内接矩形，最小外接矩形，根据标记物判断尺寸。 标记物必须位于左上角
"""

from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

def order_points(pts):
    # pts为轮廓坐标
    # 列表中存储元素分别为左上角，右上角，右下角和左下角
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


# Function to show array of images (intermediate results)
def show_images(images):
	for i, img in enumerate(images):
		cv2.imshow("image_" + str(i), img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



img_path = "images/critical0.jpg"

# Read image and preprocess
image = cv2.imread(img_path)

# 转黑白
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
# 高斯模糊，不然canny查不到轮廓
blur = cv2.GaussianBlur(gray, (9, 9), 0)
cv2.imshow("blur", blur)


edged = cv2.Canny(blur, 50, 100)


edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
cv2.imshow("Canny", blur)
#show_images([blur, edged])

# Find contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Sort contours from left to right as leftmost contour is reference object
(cnts, _) = contours.sort_contours(cnts)

# Remove contours which are not large enough
cnts = [x for x in cnts if cv2.contourArea(x) > 400]

cv2.drawContours(image, cnts, -1, (0,250,0), 2)

#show_images([image, edged])
#print(len(cnts))

#Find corners 
corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)
for i in corners :
	x,y = i.ravel()
	cv2.circle(image,(x,y),3,255,-1)
cv2.imshow("corners",image)


c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
rect = order_points(c.reshape(c.shape[0], 2))
print(rect)
xs = [i[0] for i in rect]
ys = [i[1] for i in rect]
xs.sort()
ys.sort()
#内接矩形的坐标为
#print(xs,ys)
#print(xs[1],xs[2],ys[1],ys[2])
print(type(rect[0]))
print(rect[0])
image = cv2.circle(image,tuple(rect[0]),radius=6, color=(255,0,255),thickness=2)
image = cv2.circle(image,tuple(rect[1]),radius=6, color=(255,0,255),thickness=2)
image = cv2.circle(image,tuple(rect[2]),radius=6, color=(255,0,255),thickness=2)
image = cv2.circle(image,tuple(rect[3]),radius=6, color=(255,0,255),thickness=2)
# Reference object dimensions
# Here for reference I have used a 2cm x 2cm square
ref_object = cnts[0]
box = cv2.minAreaRect(ref_object)
box = cv2.boxPoints(box)
box = np.array(box, dtype="int")
box = perspective.order_points(box)
(tl, tr, br, bl) = box
dist_in_pixel = euclidean(tl, tr)
dist_in_cm = 2
pixel_per_cm = dist_in_pixel/dist_in_cm

# Draw remaining contours
for cnt in cnts:
	print(cnt[0])
	print(cnt[0][0][0])
	print(cnt[0][0][1])
	leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
	rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
	topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
	bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
	image = cv2.circle(image,leftmost,radius=6, color=(255,255,255),thickness=2)
	image = cv2.circle(image,rightmost,radius=6, color=(255,255,255),thickness=2)
	image = cv2.circle(image,topmost,radius=6, color=(255,255,255),thickness=2)
	image = cv2.circle(image,bottommost,radius=6, color=(255,255,255),thickness=2)
	#image = cv2.circle(image,cnt[0],radius=6, color=(255,255,255),thickness=2)
	cv2.imshow("circle",image)


	box = cv2.minAreaRect(cnt)
	box = cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	box = perspective.order_points(box)

	
	(tl, tr, br, bl) = box
	cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
	mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
	mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
	wid = euclidean(tl, tr)/pixel_per_cm
	ht = euclidean(tr, br)/pixel_per_cm
	cv2.putText(image, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
	cv2.putText(image, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

show_images([image])