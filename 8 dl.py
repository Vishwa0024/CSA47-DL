import cv2
from skimage import data
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.pyplot as plt

image = data.astronaut()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

filtered_image = cv2.medianBlur(gray, 5)

thresh = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

closed = closing(thresh, square(3))

label_image = label(closed)
image_label_overlay = label2rgb(label_image, image=image)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)

for region in regionprops(label_image):
    minr, minc, maxr, maxc = region.bbox
    rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                         fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

plt.title('Segmentation Results')
plt.show()
