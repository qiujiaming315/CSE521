import os
import image_resize
import numpy as np
import cv2
import synthesize


def config_define():
    obj_list = ['hot_pad', 'pan', 'oatmeal', 'bowl', 'measure_cup', 'measure_spoon', 'metal_spoon', 'salt',
                'stirring_spoon', 'timer']

    resize_param_dict = {'bowl': 25, 'hot_pad': 20, 'measure_cup': 20, 'measure_spoon': 20, 'metal_spoon': 18,
                         'oatmeal': 27, 'pan': 35, 'salt': 20, 'stirring_spoon': 26, 'timer': 20}

    class_label = {'bowl': 0, 'hot_pad': 1, 'measure_cup': 2, 'measure_spoon': 3, 'metal_spoon': 4,
                   'oatmeal': 5, 'pan': 6, 'salt': 7, 'stirring_spoon': 8, 'timer': 9}

    raw_data_dir = '/data/hangyue/521_data/raw'
    return obj_list, resize_param_dict, class_label, raw_data_dir


def dataset_generate(obj_list, resize_param_dict, class_label, raw_data_dir):
    output_image_dir = os.path.join('/data/hangyue/521_data', 'generated')
    output_annotation_dir = os.path.join('/data/hangyue/521_data', 'annotations')

    for num_sample in range(100):
        obj_img_list = []
        obj_label_list = []
        obj_select = np.random.rand(len(obj_list))
        obj_select = (obj_select > 0.3).astype(bool)

        for item_i in range(len(obj_list)):
            if obj_select[item_i]:
                obj_cur = obj_list[item_i]
                obj_dir_cur = os.path.join(raw_data_dir, obj_cur)
                obj_dir_list = os.listdir(obj_dir_cur)
                obj_random = np.random.choice(obj_dir_list)
                image_cur = cv2.imread(os.path.join(obj_dir_cur, obj_random), cv2.IMREAD_UNCHANGED)

                image_crop = image_resize.crop_non_transparent(image_cur)
                # cv2.imwrite(os.path.join('/data/hangyue/521_data', 'gen_test', 'test{}.jpg'.format(1)), image_crop)
                image_crop = image_resize.resize_image(image_crop, resize_param_dict[obj_cur])
                # cv2.imwrite(os.path.join('/data/hangyue/521_data', 'gen_test', 'test{}.jpg'.format(2)), image_crop)
                angle_cur = np.random.choice(np.arange(360))
                image_crop = image_resize.image_rotate(image_crop, angle_cur)
                # cv2.imwrite(os.path.join('/data/hangyue/521_data', 'gen_test', 'test{}.jpg'.format(3)), image_crop)
                obj_class_ind = class_label[obj_cur]

                obj_img_list.append(image_crop)
                obj_label_list.append(obj_class_ind)

            else:
                continue

        background_dir = os.path.join(raw_data_dir, 'background')
        background_list = os.listdir(background_dir)
        back_cur = np.random.choice(background_list)
        back_img = cv2.imread(os.path.join(background_dir, back_cur), cv2.IMREAD_UNCHANGED)
        back_img = image_resize.resize_image(back_img, 100, background=True)

        synthesize.img_synthesize(back_img, obj_img_list, obj_label_list,
                                  os.path.join('/data/hangyue/521_data', 'gen_test'),
                                  os.path.join('/data/hangyue/521_data', 'gen_test'), num_sample)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    object_list, resize_param, class_dict, raw_dir = config_define()
    dataset_generate(object_list, resize_param, class_dict, raw_dir)
