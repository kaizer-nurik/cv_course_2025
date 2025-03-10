import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread("./data/the_ambassadors_skull_transformed.jpg")
img2 = cv2.imread("./data/the_ambassadors.jpg")

original_image_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
distorted_image_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image_rgb)
plt.title('Оригинальное изображение')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(distorted_image_rgb)
plt.title('Искаженное изображение')
plt.axis('off')

# plt.show()

print("Выберите 4 контрольные точки на оригинальном изображении")
original_points = plt.ginput(4)  

print("Выберите 4 контрольные точки на искажённом изображении")
distorted_points = plt.ginput(4)  

original_points = np.array(original_points, dtype=np.float32)
distorted_points = np.array(distorted_points, dtype=np.float32)
print(original_points,distorted_points)
matrix = cv2.getPerspectiveTransform(distorted_points, original_points)

height, width = img1.shape[:2]
print(matrix)
corrected_image = cv2.warpPerspective(img2, matrix, (width, height))

# Отображаем результат
corrected_image_rgb = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)
plt.imshow(corrected_image_rgb)
plt.title('Исправленное изображение')
plt.axis('off')
plt.show()


