import cv2
import numpy as np

# Object resizing parameters.
object_resize_noise = 0.1
object_resize_min_percent = 3


def resize_image(original_image, scale_percent, random_state, background=False):
    # Load the original image
    # original_image = cv2.imread(image_path)

    # Calculate the new dimensions
    max_dim = max(original_image.shape[0], original_image.shape[1])
    if background:
        resize_ratio = min(original_image.shape[0] / 480, original_image.shape[1] / 640)
    else:
        scale_percent *= 1 + ((random_state.rand() - 0.5) * 2 * object_resize_noise)
        scale_percent = max(object_resize_min_percent, scale_percent)
        max_dim_resize = int(640 * scale_percent / 100)
        resize_ratio = max_dim / max_dim_resize
    height, width = int(original_image.shape[0] / resize_ratio), int(original_image.shape[1] / resize_ratio)
    new_dimensions = (width, height)

    # Resize the image
    resized_image = cv2.resize(original_image, new_dimensions, interpolation=cv2.INTER_AREA)
    # Crop background images.
    if background:
        height, width = max(height, 480), max(width, 640)
        x_start = 0 if width == 640 else random_state.randint(width - 640)
        y_start = 0 if height == 480 else random_state.randint(height - 480)
        resized_image = resized_image[y_start:y_start + 480, x_start:x_start + 640]

    # Save the scaled image
    # cv2.imwrite(output_path, resized_image)

    # If you want to display the image (for testing purposes), uncomment the following lines
    # cv2.imshow("Resized image", resized_image)
    # cv2.waitKey(0)  # waits until a key is pressed
    # cv2.destroyAllWindows()  # destroys the window showing the image
    return resized_image


# Example usage: Resize the image to 50% of its original dimensions
# resize_image("path_to_your_image.jpg", "output_image.jpg", 50)


def crop_non_transparent(image):
    # Step 1: Load the image
    # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if the image was successfully loaded
    # if image is None:
    #     raise FileNotFoundError(f"Could not load image at {image_path}")

    # Check if the image has an alpha (transparency) channel
    if image.shape[2] < 4:
        raise TypeError("Image does not have an alpha channel (transparency)")

    # Step 2: Create a binary mask of the alpha channel
    alpha_channel = image[:, :, 3]
    _, binary_mask = cv2.threshold(alpha_channel, thresh=1, maxval=255, type=cv2.THRESH_BINARY)

    # Step 3: Find contours and the bounding box enclosing the non-transparent pixels
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If there are no contours, it means the entire image is transparent or empty.
    if not contours:
        raise ValueError("The image does not contain non-transparent pixels")

    x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # Step 4: Crop the image based on the bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Step 5: Save or display the result
    # cv2.imwrite(output_path, cropped_image)

    # Optionally display the image (for testing purposes)
    # cv2.imshow("Cropped Image", cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return cropped_image  # return the dimensions of the cropped area


def image_rotate(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    scale = 1.0

    # Get the rotation matrix using cv2.getRotationMatrix2D
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    rotation_matrix[0, 2] += bound_w / 2 - center[0]
    rotation_matrix[1, 2] += bound_h / 2 - center[1]

    # Rotate the entire image (including alpha channel)
    rotated_img = cv2.warpAffine(image, rotation_matrix, (bound_w, bound_h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # If the image has an alpha channel, split and merge is required for transparency
    if image.shape[2] == 4:
        # Split the image into the BGR and Alpha channels
        b, g, r, a = cv2.split(rotated_img)

        # Merge the channels including alpha
        rotated_img = cv2.merge([b, g, r, a])

    # cv2.imshow("rotated Image", rotated_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return rotated_img
