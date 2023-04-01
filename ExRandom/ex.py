import cv2
import numpy as np
import sys

# carregar imagens

img_left, img_right = cv2.imread(str(sys.argv[1])), cv2.imread(str(sys.argv[2]))

# converter imagens para escala de cinza
gray_left, gray_right = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# calcular mapa de disparidade
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(gray_left, gray_right)

# normalizar o mapa de disparidade
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# criar anaglifo
height, width = img_left.shape[:2]
anaglyph = np.zeros((height, width, 3), dtype=np.uint8)
anaglyph[:, :, 0], anaglyph[:, :, 1], anaglyph[:, :, 2] = img_right[:, :, 2], img_left[:, :, 1], img_left[:, :, 0]

# exibir anaglifo
cv2.imshow("Anaglyph", anaglyph)
cv2.waitKey(0)

# salvar anaglifo
cv2.imwrite("anaglyph.jpg", anaglyph)
