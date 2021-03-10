import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def pencilSketch(image):
    imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    imageGauss = cv.GaussianBlur(imageGray, (3, 3), 0, 0)
    imageLap = np.float32(imageGauss)
    imageLap = cv.Laplacian(imageGauss, cv.CV_32F, ksize=5, scale=1, delta=0)
    cv.normalize(imageLap, dst=imageLap, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    imageLap = imageLap * 255
    imageLap = np.uint8(imageLap)
    ret, sketch = cv.threshold(imageLap, 150, 255, cv.THRESH_BINARY_INV)
    pencilSketchImage = cv.merge((sketch, sketch, sketch))
    return pencilSketchImage

def cartoonify(image):
    imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    imageGauss = cv.GaussianBlur(imageGray, (3, 3), 0, 0)
    imageLap = np.float32(imageGauss)
    imageLap = cv.Laplacian(imageGauss, cv.CV_32F, ksize=5, scale=1, delta=0)
    cv.normalize(imageLap, dst=imageLap, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    ret, imEdges = cv.threshold(imageLap, 150/255, 1, cv.THRESH_BINARY_INV)
    
    imageBilFilter = cv.bilateralFilter(image, 9, 150, 150)
    imageLessColor = imageBilFilter / 17
    imageLessColor = np.uint8(imageLessColor)
    imageLessColor = 17 * imageLessColor

    imageLessColor = np.float32(imageLessColor)
    imageLessColor = imageLessColor / 255

    imEdgesMerged = cv.merge((imEdges, imEdges, imEdges))
    cartoonImage = cv.multiply(imageLessColor, imEdgesMerged)

    cartoonImage = cartoonImage * 255
    cartoonImage = np.uint8(cartoonImage)
    return cartoonImage
    
def main():
    winName = "cartoon"
    img = cv.imread("zdj/person.jpg", cv.IMREAD_COLOR)
    imgClone = img.copy()
    
    img = pencilSketch(img)
    
    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    cv.imshow(winName, img)
    cv.waitKey()

    img = cartoonify(imgClone)

    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    cv.imshow(winName, img)
    cv.waitKey()


if (__name__ == "__main__"):
    main()
    
