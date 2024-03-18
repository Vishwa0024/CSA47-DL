import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import data
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.color import rgb2gray

image = data.coins()

if len(image.shape) > 2 and image.shape[2] > 1:
    gray = rgb2gray(image)
else:
    gray = image

distance = ndi.distance_transform_edt(gray)
peaks = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=gray)

markers = ndi.label(peaks)[0]
labels = watershed(-distance, markers, mask=gray)

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(gray, cmap=plt.cm.gray)
ax[0].set_title('Original Image')

ax[1].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[1].set_title('Segmented Image')

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()
