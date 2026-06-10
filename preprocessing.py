#%%
# ==========================================
# IMPORT LIBRARIES
# ==========================================

import cv2
import numpy as np


# ==========================================
# IMAGE PREPROCESSING PIPELINE
# ==========================================

class ImagePreprocessingPipeline:

    def __init__(self, img_size=224):
        self.img_size = img_size

    def transform(self, image):
        """
        Apply preprocessing to a single image

        Steps:
        1. Resize
        2. Convert to float32
        3. Normalize
        """

        # Resize image
        image = cv2.resize(
            image,
            (self.img_size, self.img_size)
        )

        # Convert datatype
        image = image.astype(np.float32)

        # Normalize
        image = image / 255.0

        return image

    def transform_batch(self, images):
        """
        Apply preprocessing to all images
        """

        processed_images = []

        for image in images:

            processed_image = self.transform(image)

            processed_images.append(
                processed_image
            )

        return np.array(processed_images)
# %%
