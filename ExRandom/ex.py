import cv2
import numpy as np
import sys

# carregar imagens
img_left, img_right = cv2.imread("./imagem3L.png"), cv2.imread("./imagem3R.png")

# criar anaglifo
height, width = img_left.shape[:2]
anaglyph = np.zeros((height, width, 3), dtype=np.uint8)
anaglyph[:, :, 0], anaglyph[:, :, 1], anaglyph[:, :, 2] = img_left[:, :, 1], img_left[:, :, 2], img_right[:, :, 0]

# exibir anaglifo
cv2.imshow("Anaglyph", anaglyph)
cv2.waitKey(0)

# salvar anaglifo
cv2.imwrite("anaglyph.jpg", anaglyph)
