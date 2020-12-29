import cv2
import numpy as np
import utlis

##################################################
path = "4.jpg"
widthimg = 700
heightimg = 700
questions = 5
choices = 5
##################################################

img = cv2.imread(path)

 # procesando
newimg = cv2.resize(img,(widthimg,heightimg))

imgContours = newimg.copy()
imgBiggestContours = newimg.copy()

imgGray = cv2.cvtColor(newimg,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny = cv2.Canny(imgBlur,10,50)

#Encontrar todos los contornos
contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours,contours,-1,(0,255,0),10)

#Encotrar Rectangulos
rectCon = utlis.rectContour(contours)
biggestContour = utlis.getCornerPoints(rectCon[0])
    #print(biggestContour.shape)
gradePoints = utlis.getCornerPoints(rectCon[1]) #en nuestra imagen cojemos el rectangulo 2 del array por que es l 3ro de tamaño
#print(biggestContour)

if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),20)
    cv2.drawContours(imgBiggestContours,gradePoints,-1,(255,0,0),20)

    biggestContour = utlis.reorder(biggestContour)
    gradePoints = utlis.reorder(gradePoints)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0,0],[widthimg,0],[0,heightimg],[widthimg,heightimg]])
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    imgWarpColores = cv2.warpPerspective(newimg,matrix,(widthimg,heightimg))

    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0,0],[325,0],[0,150],[325,150]])
    matrixG = cv2.getPerspectiveTransform(ptG1,ptG2)
    imgGradeDisplay = cv2.warpPerspective(newimg,matrixG,(325,150))
    #cv2.imshow("Puntos",imgGradeDisplay)    

    #Aplicamos umbral
    imgWarpGray = cv2.cvtColor(imgWarpColores,cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray,170,255,cv2.THRESH_BINARY_INV)[1]

    boxes = utlis.splitBoxes(imgThresh)
    #cv2.imshow("Test",boxes[2])

    #No obtener valores de píxeles de ceros de cada cuadro
    myPixelVal = np.zeros((questions,choices))
    countC = 0
    countR = 0

    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if (countC == choices):countR += 1; countC = 0
    print(myPixelVal)

imgBlank = np.zeros_like(newimg)
imageArray = ([newimg,imgGray,imgBlur,imgCanny],
            [imgContours,imgBiggestContours,imgWarpColores,imgThresh])
imgStacked = utlis.stackImages(imageArray,0.4)

cv2.imshow("Proyecto Final",imgStacked)
cv2.waitKey(0)