import cv2
import numpy as np
import os
import random

# Parameters that control overlapping objects.
max_overlap = 0.2
container_objects = [1]
max_attempts = 10


# Function to overlay transparent objects on background
def overlay_transparent(background_img, overlay_img, x, y):
    # overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGRA2RGBA)
    h, w, _ = overlay_img.shape
    overlay_alpha = overlay_img[:, :, 3:] / 255.0
    background_alpha = 1.0 - overlay_alpha

    for c in range(3):
        color = np.expand_dims(background_img[y:y + h, x:x + w, c], axis=-1) * (background_alpha) \
                + np.expand_dims(overlay_img[:, :, c], axis=-1) * (overlay_alpha)
        background_img[y:y + h, x:x + w, c] = color.squeeze()

    return background_img


def calculate_intersection_area(box1, box2):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box1['x'], box2['x'])
    yA = max(box1['y'], box2['y'])
    xB = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
    yB = min(box1['y'] + box1['height'], box2['y'] + box2['height'])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea


def is_acceptable_overlap(box1, box2):
    interArea = calculate_intersection_area(box1, box2)
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']

    overlap_with_box1 = interArea / float(box1_area)
    overlap_with_box2 = interArea / float(box2_area)

    # Check if overlap is within the acceptable range for either box
    return (overlap_with_box1 <= max_overlap) or (overlap_with_box2 <= max_overlap)


def place_object(objects, new_object):
    for obj in objects:
        if is_acceptable_overlap(obj, new_object):
            # Overlap is acceptable
            continue
        else:
            # Overlap is not acceptable
            return False
    # objects.append(new_object)  # Place the object
    return True


def img_synthesize(background, obj_list, class_list, output_images_dir, output_labels_dir, sample_ind):
    img_height, img_width, _ = background.shape
    num_objects = len(obj_list)
    annotations = []
    bound_hist = []

    for i, obj_cur in enumerate(obj_list):
        # Random position for object
        for j in range(max_attempts):
            x = random.randint(0, img_width - obj_cur.shape[1])
            y = random.randint(0, img_height - obj_cur.shape[0])

            # YOLOv5 format for bounding box: <class> <x_center> <y_center> <width> <height>
            obj_height, obj_width, _ = obj_cur.shape
            x_center = x + obj_width / 2
            y_center = y + obj_height / 2

            cur_obj_bound = {'x': x, 'y': y, 'width': obj_width, 'height': obj_width}
            if place_object(bound_hist, cur_obj_bound):
                overlay_transparent(background, obj_cur, x, y)
                if class_list[i] not in container_objects:
                    # assume class 'hot pad', 'pot', and 'bowl' can be overlapped
                    bound_hist.append(cur_obj_bound)

                # Normalize coordinates by image dimensions
                x_center /= img_width
                y_center /= img_height
                obj_width /= img_width
                obj_height /= img_height

                annotations.append(f"{class_list[i]} {x_center} {y_center} {obj_width} {obj_height}")
                break

    # Save the synthetic image
    output_image_path = os.path.join(output_images_dir, f"image_{sample_ind}.jpg")
    cv2.imwrite(output_image_path, background)

    # Save annotations for this image
    output_label_path = os.path.join(output_labels_dir, f"image_{sample_ind}.txt")
    with open(output_label_path, 'w') as f:
        for annotation in annotations:
            f.write("%s\n" % annotation)
