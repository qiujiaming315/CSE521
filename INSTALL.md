### Synthetic Dataset

The synthetic images are generated based on raw images of common items in kitchen collected from the Internet. We use images of kitchen countertops as background and items such as bowls and oatmeal as foreground objects. You can download our [collection](https://drive.google.com/file/d/1djoCXzBf3hUa2UWuIM4DuAvyDc6tiTu_/view?usp=drive_link) of such raw images.

### Real-Time Detection with YOLOv5

You should first clone the public [YOLOv5](https://github.com/ultralytics/yolov5) repo as a backbone of the object detector. Follow the instructions from the YOLOv5 webpage to install all the required libraries. You can then directly copy the files in our ``yolo_patch`` folder (preserve the hierarchy we provide) to overwrite the files in your cloned YOLOv5 repo.
