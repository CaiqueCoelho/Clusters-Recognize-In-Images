#Alunos:
#Caique de Paula Figueiredo Coelho
#Libero Passador Neto
#Observacao: o programa nao funcionou bem para a imagem bois2.jpg

print(__doc__)

import time as time

import numpy as np
import scipy as sp
import sys
import cv2
import PIL.ImageOps
import matplotlib.pyplot as plt

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version
from PIL import Image

#if sp_version < (0, 12):
#    raise SkipTest("Skipping because SciPy version earlier than 0.12.0 and "
#                   "thus does not include the scipy.misc.face() image.")

#funcao transforma imagem de rgb em cinza
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


#funcao inverte cores da imagem(negativo)
def inverte(imagem):
    return (255-imagem)

###############################################################################
# load data


if (len(sys.argv) != 2 ):
    print "Informe a imagem de entrada junto, exemplo: python %s bois1.jpg"%(sys.argv[0])
    sys.exit(0)

face = sp.misc.imread(sys.argv[1])

#face = sp.misc.imread(sys.argv[1], 0)
#face = cv2.imread(sys.argv[1])
#face = Image.open(sys.argv[1])

#face = rgb2gray(face)

#face = cv2.bilateralFilter(face, 5, 175, 175)
#cv2.imshow('Bilateral', face)
#cv2.waitKey(0)

#face = cv2.GaussianBlur(face, (5, 5), 0)

#face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)	

#face = cv2.bilateralFilter(face, 5, 175, 175)
#cv2.imshow('Bilateral', face)
#cv2.waitKey(0)

#face = cv2.Canny(face, 75, 200)
#cv2.imshow('Edge', face)
#cv2.waitKey(0)
#face = PIL.ImageOps.invert(face)

#face = inverte(face)


# Resize it to 50% of the original size to speed up the processing
face = sp.misc.imresize(face, 0.20) / 255.

X = np.reshape(face, (-1, 1))


###############################################################################
# Define the structure A of the data. Pixels connected to their neighbors.
connectivity = grid_to_graph(*face.shape)


###############################################################################
# Compute clustering
print("Compute structured hierarchical clustering...")

st = time.time()

n_clusters = 2  # number of regions

ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=connectivity)
ward.fit(X)
label = np.reshape(ward.labels_, face.shape)
###############################################################################



#Faz os contornos nos bois pretos
#plt.imshow(rgb2gray(face),cmap='gray')
for l in range(n_clusters):
    m =  label == l
    plt.contour(m[:,:,0], contours=1,
                colors=[plt.cm.spectral(l / float(n_clusters)), ])
plt.xticks(())
plt.yticks(())
#plt.show()

#Faz os contornos nos bois brancos e marrons
#plt.imshow(rgb2gray(face),cmap='gray')
for l in range(n_clusters):
    m =  label == l
    plt.contour(m[:,:,2], contours=1,
                colors=[plt.cm.spectral(l / float(n_clusters)), ])
plt.xticks(())
plt.yticks(())
#salvando imagem com os contornos nos boins brancos, marrons e pretos
plt.savefig('out_contornos.jpg')
plt.show()

#Carrega novamente imagem de entrada para encontrar possiveis sombras
sombras = sp.misc.imread(sys.argv[1])

# Resize it to 50% of the original size to speed up the processing
sombras = sp.misc.imresize(sombras, 0.20) / 255.

Y = np.reshape(sombras, (-1, 1))


###############################################################################
# Define the structure A of the data. Pixels connected to their neighbors.
connectivity = grid_to_graph(*sombras.shape)

###############################################################################

# Compute clustering sombras
#print("Compute structured hierarchical clustering...")
st = time.time()
n_clusters = 2  # number of regions
ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=connectivity)
ward.fit(Y)
label = np.reshape(ward.labels_, sombras.shape)



###########################################################################

#Faz o contorno das possiveis sombras
for l in range(n_clusters):
    m =  label == l
    plt.contour(m[:,:,0], contours=1,
                colors=[plt.cm.spectral(l / float(n_clusters)), ])
plt.xticks(())
plt.yticks(())
#plt.show()

#salva imagem com os contornos das possiveis sombras
plt.savefig('sombras_out.jpg')

#carrega imagem com os contornos das possiveis sombras e coloca a imagem em escala cinza
sombras_in = cv2.imread('sombras_out.jpg', 0)

#carrega a imagem com os contornos dos bois e coloca a imagem em escala cinza
image = cv2.imread('out_contornos.jpg', 0)

#Tamanho minimo e maximo da area do contorno para dizer se ele eh valido ou nao
threshold_area_min = 15
threshold_area_max = 150

#Tamanho minimo e maximo do perimetro do contorno para dizer se ele eh valido ou nao
threshold_per_min = 40
threshold_per_max = 85

#Realiza a busca pelos contornos dos bois
ret,thresh = cv2.threshold(image,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#Realiza a busca dos contornos das possiveis sombras
retS,threshS = cv2.threshold(sombras_in,127,255,0)
im2S, contoursS, hierarchyS = cv2.findContours(threshS,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#variaveis contadoras para bois usando funcao de area/per e para sombras
bois_area = 0
bois_per = 0
sombras = 0;

#conta possivel quantidade de sombras unsando funcao arcLenght
for cnt in contoursS:        
    area = cv2.arcLength(cnt,True)     
    if area > threshold_per_min and area < threshold_per_max:                   
        sombras += 1

#Filtro de tamanho da area do contorno usando funcao contourArea para contar contornos validos
for cnt in contours:        
    area = cv2.contourArea(cnt)         
    if area > threshold_area_min and area < threshold_area_max:                   
        if area > 140 and area <= 150:
            bois_area += 2

        #elif area > 160 and area < threshold_area_max:
        #    bois_area += 3

        else:
            bois_area += 1

#Filtro de tamanho de perimetro do contorno usando funcao arcLenght para contar contornos validos
for cnt in contours:        
    perimetro = cv2.arcLength(cnt,True)        
    if perimetro > threshold_per_min and perimetro < threshold_per_max:                   
        #if perimetro > 80:
        #    bois_per += 2

        #elif perimetro > 80 and perimetro < threshold_per_max:
        #    bois_per += 3

        #else:
            bois_per += 1

#teste = 0
#while hierarchy.all():
    #cv.DrawContours(img, contours, colours[i], colours[i], 0, thickness=-1)
    #i = (i+1) % len(colours)
    #teste += 1
    #hierarchy = hierarchy.h_next() # go to next contour
#print teste
print "\n\n"
print"#########################################################################################"
print "ATENCAO, SE A IMAGEM CONTEM SOMBRAS DOS BOIS, A SOMBRA EH CONTADA COMO UM BOI PRETO!"
print"#########################################################################################"

print "\n"
print "Possivel quantidade de sombras ou bois pretos ou bois com manchas pretas: " + str(sombras)
print "\n"

print"####################################################################################################################################"

print "\n"

print "CONSIDERANDO POSSIVEIS SOMBRAS COMO BOIS"

print "\n"

print "Temos " + str(bois_area) + " bois, usando a funcao contourArea para contar"
print "Temos " + str(bois_per) + " bois, usando a funcao arcLenght para contar"
print "Temos ente " + str(bois_area) + "~" + str(bois_per) + " bois nessa imagem"
print "Alta possibilidade de termos exatos " + str((bois_area + bois_per)/2) + " bois nessa imagem"
print "Margem de erro para mais ou para menos bois eh de: " + str(abs(bois_area - bois_per)) + " bois"

print "\n"

print"####################################################################################################################################"

print"\n"

print "DESCONSIDERANDO POSSIVEIS SOMBRAS COMO BOIS E CONSEQUENTEMENTE BOIS PRETOS E BOIS COM MANCHAS PRETAS"

print "\n"

print "Temos " + str(bois_area - sombras) + " bois, usando a funcao contourArea para contar"
print "Temos " + str(bois_per - sombras) + " bois, usando a funcao arcLenght para contar"
print "Temos ente " + str(bois_area - sombras) + "~" + str(bois_per - sombras) + " bois nessa imagem"
print "Alta possibilidade de termos exatos " + str( ( (bois_area - sombras) + (bois_per - sombras) ) /2) + " bois nessa imagem"
print "Margem de erro para mais ou para menos bois eh de: " + str( abs( (bois_area - sombras) - (bois_per - sombras) ) ) +" bois"

print "\n"

print"####################################################################################################################################"