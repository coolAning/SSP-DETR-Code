_base_ = ['DQ_5scale_pcb.py']

# CPH (COCO detection) config.
#
# Dataset layout accepted by dataset_file='coco' in this repo:
#   - train/ or images/train/
#   - val/ or test/ or images/val/ or images/test/
#   - annotations/instances_train.json
#   - annotations/instances_val.json or annotations/instances_test.json

# CPH has 5 categories with ids 0..4 in the provided annotations.
num_classes = 5
dn_labelbook_size = 5

# Enable density-biased matching cost by default for the CPH full setting.
set_cost_pmap = 0.05
cost_pmap_eps = 1e-6
cost_pmap_warmup_epochs = 4
cost_pmap_q_mode = 'localmax'
cost_pmap_localmax_r = 1
