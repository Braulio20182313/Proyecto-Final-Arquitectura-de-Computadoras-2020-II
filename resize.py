import cv2
import numpy as np
import utlis

##################################################
path = "1.jpg"
widthimg = 700
heightimg = 700
questions = 5
choices = 5
ans = [1,2,0,1,4] #respuestas correctas

webCamFeed = True
cameraNo = 1
##################################################

cap = cv2.VideoCapture(cameraNo)
cap.set(10,150)

while True:
    if webCamFeed: success, img = cap.read()
    else: img = cv2.imread(path)


    # procesando
    newimg = cv2.resize(img,(widthimg,heightimg))

    imgContours = newimg.copy()
    imgFinal = newimg.copy()
    imgBiggestContours = newimg.copy()

    imgGray = cv2.cvtColor(newimg,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,10,50)

    try:
            
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
            imgThresh = cv2.threshold(imgWarpGray,110,255,cv2.THRESH_BINARY_INV)[1]

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
            #print(myPixelVal)


            #Encontrar valores de índice de la marca
            myIndex = []
            for x in range (0,questions):
                arr = myPixelVal[x]
                #print("arr",arr)
                myIndexVal = np.where(arr == np.amax(arr))
                #print(myIndexVal)
                myIndex.append(myIndexVal[0][0])
            #print(myIndex)

            #Calificación
            grading = []
            for x in range (0,questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else: grading.append(0)
            #print (grading)
            score = (sum(grading)/questions)*100 #calificacion final en porcentaje
            print(score)

            # MOSTRANDO RESPUESTAS
            imgResult = imgWarpColores.copy()
            imgResult = utlis.showAnswer(imgResult,myIndex,grading,ans,questions,choices)

            imRawDrawing = np.zeros_like(imgWarpColores)
            imRawDrawing = utlis.showAnswer(imRawDrawing,myIndex,grading,ans,questions,choices)

            invMatrix = cv2.getPerspectiveTransform(pt2,pt1)
            imgInvWarp = cv2.warpPerspective(imRawDrawing,invMatrix,(widthimg,heightimg))

            imgRawGrade = np.zeros_like(imgGradeDisplay)
            cv2.putText(imgRawGrade,str(int(score))+"%",(60,100),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,3,(0,255,255),10)
            invMatrixG = cv2.getPerspectiveTransform(ptG2,ptG1)
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade,invMatrixG,(widthimg,heightimg))

            imgFinal = cv2.addWeighted(imgFinal,1,imgInvWarp,1,0)
            imgFinal = cv2.addWeighted(imgFinal,1,imgInvGradeDisplay,1,0)

        imgBlank = np.zeros_like(newimg)
        #matriz de imagenes para visualizacion
        imageArray = ([newimg,imgGray,imgBlur,imgCanny],
                    [imgContours,imgBiggestContours,imgWarpColores,imgThresh],
                    [imgResult,imRawDrawing,imgInvWarp,imgFinal])
    except:
        imgBlank = np.zeros_like(newimg)
        imageArray = ([newimg,imgGray,imgBlur,imgCanny],
                  [imgBlank,imgBlank,imgBlank,imgBlank],
                  [imgBlank,imgBlank,imgBlank,imgBlank])
    lables = [["Original","Gris","difuminado","Canny"],
            ["Contornos", "Contornos Gr","Forma","limites"],
            ["Resultado", "previo","deformación inversa", "imagen Final"]]
    imgStacked = utlis.stackImages(imageArray,0.3,lables)

    cv2.imshow("Imagen Final Presentacion",imgFinal)
    cv2.imshow("Proyecto Final",imgStacked)
    if cv2.waitKey(1) & 0xFF == ord ('s'):
        cv2.imwrite("resultado_final.jpg",imgFinal)
        cv2.waitKey(300)