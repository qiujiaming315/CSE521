### Synthetic Dataset

Once you have downloaded/collected the set of raw images of kitchen items, you can run our `main.py` to generate as many synthetic images as you want. An example run looks like:
```script
# Generate 1000 synthetic images.
# IMG_PATH: path to your raw images.
# OUT_PATH: path to save the generated images and annotations.
python main.py IMG_PATH OUT_PATH --img-num 1000
```
You may run ``python main.py -h`` to view the full description of available command line arguments.

### Real-Time Detection with YOLOv5

#### Training

You should first train an YOLOv5 object detector using YOLOv5's ``train.py``  based on our custom smart kitchen dataset, which consists of the synthetic images generated in the previous step. To fine-tune a model pre-trained on [COCO](https://cocodataset.org/#home) for 300 epochs, run:
```script
# You should nevigate to the cloned YOLOv5 folder to execute this command.
# Please first configure a yaml file to point to your custom dataset.
# YAML_PTAH: path to your dataset configuration file.
python train.py --data YAML_PTAH --weights yolov5n.pt --epochs 300
```

#### Detection

You can run the ``detect.py`` from the YOLOv5 repo (with the patch inserted) to perform real-time object detection. Optionally, you can convert the trained YOLOv5 model into ``onnx`` format for faster inference speed on Raspberry Pi. An example run to perform real-time detection on video stream captured by the webcam looks like:
```script
# You should nevigate to the cloned YOLOv5 folder to execute this command.
# --source 0 refers to the webcam.
# The three thresholds refers to thresholds for open-world detection. 
# WTS_PTAH: path to the trained model weights.
# DET_PTAH: path to save the detection results.
python detect.py --weights WTS_PTAH --source 0 --det-path DET_PTAH --conf-thres 0.25 --dist-thres 0.1 obj-thres 0.5
```

Since you are running the modified version of ``detect.py`` from our ``yolo_patch``, the detection results will be saved to a ``.txt`` file. You can next run the ``voice.py`` (again from the ``yolo_patch``) in a separate thread to start the AI-voice service:
```script
# You should nevigate to the cloned YOLOv5 folder to execute this command.
# DET_PATH: path to the collected detection results.
# VOICE_PATH: path to save generated audio file.
python voice.py DET_PATH VOICE_PATH
```
