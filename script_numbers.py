# en este script la idea es obtener un array con los numeros de interes en una imagen

# exporto librerias
import numpy as np
import cv2
from skimage.metrics import structural_similarity


# defino parametros:

lower_range = np.array([0, 0,0])
upper_range = np.array([0, 0,240])

# cargo la imagen guardada
img_saved = cv2.imread("numbers.png",0)
numbers = []
for k in range(10):
    numbers.append(img_saved[:,24*k:(24*(k+1))])

# funcion que obtiene el numero de la imagen
def get_number(img):
    img = cv2.resize(img,(24,28))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    out = []
    for number in numbers:
        score,_= structural_similarity(img, number, full=True)
        out.append(score)
    return str(np.argmax(out))


# funcion que realiza el trabajo de extraer la lista donde estan los numeros de interes
def get_numbers(img):
    
    # particiono la imagen a partir de ciertos pixeles para ya que
    # los logos complican el script
    
    img1 = cv2.resize(img,(302, 542))
    img1 = img[70:,70:,:]
    
    # creo una mascara de donde extrer los numeros
    mask = cv2.inRange(cv2.cvtColor(img1,cv2.COLOR_BGR2HSV), lower_range, upper_range)
    
    # encuentro los 6 rectangulos de donde extraer los numeros
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # ordeno los rectangulos por area y extraigo 6 de ellos
    sorted_contours = sorted(contours, key = cv2.contourArea, reverse=True)

    cnts = sorted_contours[1:7]
    
    # ordeno los rectangulos por posicion 
    Bbox = [cv2.boundingRect(c) for c in cnts]
    (cnts, Bbox) = zip(*sorted(zip(cnts, Bbox),key=lambda b:b[1][1], reverse=True))

    lista = []
    
    # loop cada rectangulo (6 rectangulos)
    for i in range(len(cnts)):

        bbox = Bbox[i]
        
        # extraigo la imagen de el rectangulo donde quiero extrer los numeros y la mascara
        img2 = img1[bbox[1]:(bbox[1]+bbox[3]),bbox[0]:(bbox[0]+bbox[2])]
        mask2 = mask[bbox[1]:(bbox[1]+bbox[3]),bbox[0]:(bbox[0]+bbox[2])]

        # encuentro los 3 rectangulos de donde extraer los numeros
        contours, hierarchy = cv2.findContours(mask2,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        # ordeno los rectangulos por area y extraigo 3 de ellos
        sorted_contours = sorted(contours, key = cv2.contourArea, reverse=True)
        cnts_2 = sorted_contours[2:5]
        
        # ordeno los rectangulos por posicion 
        Bbox_2 = [cv2.boundingRect(c) for c in cnts_2]
        (cnts_2, Bbox_2) = zip(*sorted(zip(cnts_2, Bbox_2),key=lambda b:b[1][0], reverse=False))

        # comparo los numeros de cada rectangulo (3 numeros)
        number = ''

        for j in range(len(cnts_2)):
            bbox = Bbox_2[j]

            img3 = img2[bbox[1]:(bbox[1]+bbox[3]),bbox[0]:(bbox[0]+bbox[2])]

            number += get_number(img3)

        lista.append(number)
    lista.reverse()

    return lista

#################################################################################################
################## En este paso exporto una imagen y pruebo la funcion definida #################
#################################################################################################

# exporto una imagen
img = cv2.imread("panel1.png")

# imprimo los resultados
print(get_numbers(img))
