import cv2
import numpy as np
import os
import random


# Function to overlay transparent objects on background
def overlay_transparent(background_img, overlay_img, x, y):
    # overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGRA2RGBA)
    h, w, _ = overlay_img.shape
    overlay_alpha = overlay_img[:, :, 3:] / 255.0
    background_alpha = 1.0 - overlay_alpha

    for c in range(3):
        color = np.expand_dims(background_img[y:y+h, x:x+w, c], axis=-1) * (background_alpha)\
                + np.expand_dims(overlay_img[:, :, c], axis=-1) * (overlay_alpha)
        background_img[y:y+h, x:x+w, c] = color.squeeze()

    return background_img


def img_synthesize(background, obj_list, class_list, output_images_dir, output_labels_dir, sample_ind):
    img_height, img_width, _ = background.shape
    num_objects = len(obj_list)
    annotations = []
    for i, obj_cur in enumerate(obj_list):
        # Random position for object
        x = random.randint(0, img_width - obj_cur.shape[1])
        y = random.randint(0, img_height - obj_cur.shape[0])
        overlay_transparent(background, obj_cur, x, y)
        # YOLOv5 format for bounding box: <class> <x_center> <y_center> <width> <height>
        obj_height, obj_width, _ = obj_cur.shape
        x_center = x + obj_width / 2
        y_center = y + obj_height / 2

        # Normalize coordinates by image dimensions
        x_center /= img_width
        y_center /= img_height
        obj_width /= img_width
        obj_height /= img_height

        annotations.append(f"{class_list[i]} {x_center} {y_center} {obj_width} {obj_height}")

    # Save the synthetic image
    output_image_path = os.path.join(output_images_dir, f"image_{sample_ind}.jpg")
    cv2.imwrite(output_image_path, background)

    # Save annotations for this image
    output_label_path = os.path.join(output_labels_dir, f"image_{sample_ind}.txt")
    with open(output_label_path, 'w') as f:
        for annotation in annotations:
            f.write("%s\n" % annotation)

