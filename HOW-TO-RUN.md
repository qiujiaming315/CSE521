#### ORIE Calculation

Once you have collected the detection outputs of an object detector pair, you can use our `reward.py` to compute the ORIE (or ORI and DCSB) value of each image as its offloading reward. An example run looks like:
```script
# Compute ORIE using 1000 ensemble images.
# WEAK_PATH, STRONG_PATH: path to your weak/strong detector outputs.
# LABEL_PATH: path to the dataset annotations.
# SAVE_PATH: path to save the computed rewards.
python reward.py WEAK_PATH STRONG_PATH LABEL_PATH SAVE_PATH --method orie --num-ensemble 1000
```