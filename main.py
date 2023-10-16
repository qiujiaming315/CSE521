import argparse
import cv2
import os
import numpy as np
from pathlib import Path

from lib import image_resize
from lib import synthesize

"""Create synthetic images and bounding box annotations for object detection in a smart kitchen setting."""

# Declare the constants here.
required_list = ['hot_pad', 'pan', 'oatmeal', 'bowl', 'measure_cup', 'measure_spoon', 'metal_spoon', 'salt',
                 'stirring_spoon', 'timer']
common_list = ['black_pepper', 'folk', 'knife', 'peeler', 'plate', 'pot', 'scissors']
other_list = []
required_resize_dict = {'bowl': 25, 'hot_pad': 20, 'measure_cup': 20, 'measure_spoon': 20, 'metal_spoon': 18,
                        'oatmeal': 27, 'pan': 35, 'salt': 20, 'stirring_spoon': 26, 'timer': 20}
common_resize_dict = {'black_pepper': 20, 'folk': 18, 'knife': 20, 'peeler': 18, 'plate': 25,
                      'pot': 35, 'scissors': 20}
other_resize_dict = {}
required_label = {'bowl': 0, 'hot_pad': 1, 'measure_cup': 2, 'measure_spoon': 3, 'metal_spoon': 4,
                  'oatmeal': 5, 'pan': 6, 'salt': 7, 'stirring_spoon': 8, 'timer': 9}
common_label = {'black_pepper': 10, 'folk': 11, 'knife': 12, 'peeler': 13, 'plate': 14, 'pot': 15,
                'scissors': 16}
other_label = 17
img_formats = ["jpg", "jpeg", "png"]


def retrieve_image(img_dir, sample_num, random_state):
    """
    Randomly select certain number of images from a directory.
    :param img_dir: the image directory.
    :param sample_num: number of images to sample (with replacement).
    :param random_state: the random state.
    :return: a list of sampled images.
    """
    img_list = os.listdir(img_dir)
    # Collect the list of images with valid format.
    img_list = [name for name in img_list if name.split('.')[-1].lower() in img_formats]
    sample = random_state.choice(img_list, size=sample_num, replace=True)
    sample_img = [cv2.imread(os.path.join(img_dir, s), cv2.IMREAD_UNCHANGED) for s in sample]
    return sample_img


def dataset_generate(args):
    required_path = os.path.join(args.raw_dir, "required")
    common_path = os.path.join(args.raw_dir, "common")
    other_path = os.path.join(args.raw_dir, "other")
    out_img_path = os.path.join(args.out_dir, "images")
    out_label_path = os.path.join(args.out_dir, "labels")
    # Create the directories for saving the generated images and annotations.
    Path(out_img_path).mkdir(parents=True, exist_ok=True)
    Path(out_label_path).mkdir(parents=True, exist_ok=True)
    rstate = np.random.RandomState(args.seed)

    def select_objects(base_path, obj_list, obj_resize_dict, obj_label, sample_prob, reoccur_prob):
        obj_img_list, obj_label_list = [], []
        # Randomly select each type of objects.
        selection = rstate.rand(len(obj_list)) < sample_prob
        for obj_name, selected in zip(obj_list, selection):
            if selected:
                obj_path = os.path.join(base_path, obj_name)
                # Randomly select the number of objects to appear according to geometric distribution.
                num_reoccur = rstate.geometric(1 - reoccur_prob)
                obj_imgs = retrieve_image(obj_path, num_reoccur, rstate)
                for obj_img in obj_imgs:
                    # Process the selected object image.
                    obj_crop = image_resize.crop_non_transparent(obj_img)
                    obj_crop = image_resize.resize_image(obj_crop, obj_resize_dict[obj_name])
                    angle = np.random.choice(np.arange(360))
                    obj_crop = image_resize.image_rotate(obj_crop, angle)
                    obj_class = obj_label[obj_name] if isinstance(obj_label, dict) else obj_label
                    obj_img_list.append(obj_crop)
                    obj_label_list.append(obj_class)
        return obj_img_list, obj_label_list

    # Start to generate the synthetic images.
    for sample_idx in range(args.img_num):
        img_list = []
        label_list = []
        required_img_list, required_label_list = select_objects(required_path, required_list, required_resize_dict,
                                                                required_label, args.required_prob, args.reoccur_prob)
        img_list.extend(required_img_list)
        label_list.extend(required_label_list)
        common_img_list, common_label_list = select_objects(common_path, common_list, common_resize_dict,
                                                            common_label, args.common_prob, args.reoccur_prob)
        img_list.extend(common_img_list)
        label_list.extend(common_label_list)
        other_img_list, other_label_list = select_objects(other_path, other_list, other_resize_dict,
                                                          other_label, args.other_prob, args.reoccur_prob)
        img_list.extend(other_img_list)
        label_list.extend(other_label_list)
        background_dir = os.path.join(required_path, 'background')
        back_img = retrieve_image(background_dir, 1, rstate)[0]
        back_img = image_resize.resize_image(back_img, 100, background=True)
        # Generate the synthetic image.
        synthesize.img_synthesize(back_img, img_list, label_list, out_img_path, out_label_path, sample_idx)


def getargs():
    """Parse command line arguments."""
    args = argparse.ArgumentParser()
    args.add_argument('raw_dir', help="Path to the raw images (objects and backgrounds).")
    args.add_argument('out_dir', help="Path to save the generated images and annotations.")
    args.add_argument('--img-num', type=int, default=100,
                      help="Number of synthetic images to generate.")
    args.add_argument('--seed', type=int, default=None,
                      help="Random seed for reproducing.")
    args.add_argument('--required-prob', type=float, default=0.7,
                      help="Probability that each category of required items appears in the generated image.")
    args.add_argument('--common-prob', type=float, default=0.2,
                      help="Probability that each category of common distractors appears in the generated image.")
    args.add_argument('--other-prob', type=float, default=0,
                      help="Probability that each category of other distractors appears in the generated image.")
    args.add_argument('--reoccur-prob', type=float, default=0.1,
                      help="Probability that the same category of objects reappears in the generated image.")
    return args.parse_args()


if __name__ == '__main__':
    dataset_generate(getargs())
