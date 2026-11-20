_base_ = ['DQ_5scale_pcb.py']

# CPS2D (CFINet-style COCO) config
# Dataset layout expected by dataset_file='cps2d':
#   - train_images/
#   - test_images/
#   - annotations/train_labels.json
#   - annotations/test_labels.json

# CPS2D has 3 categories (ids 0..2) in the provided annotations.
num_classes = 3
