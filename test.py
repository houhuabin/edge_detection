
from utils import utils
import canny_edge_detector as ced
import cv2
imgs = utils.load_data()
#utils.visualize(imgs, 'gray')
detector = ced.cannyEdgeDetector(imgs, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
imgs_final = detector.detect()

cv2.imwrite('img_final.jpg', imgs_final[0])
#utils.visualize(imgs_final, 'gray')