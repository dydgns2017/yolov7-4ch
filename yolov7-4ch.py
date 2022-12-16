# Ipykernel
# %%
!python create_cfg_4ch.py ## create sample dataset and cfg file
# %%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
!python train.py --workers 0 --device 0 --batch-size 8 --data test_dataset/data.yaml --img 640 640 --cfg yolov7-4ch.yaml --weights 'yolov7_training.pt' --name yolov7 --hyp data/hyp.scratch.p5.yaml --epoch 30
# %%
!python test.py --data test_dataset/data.yaml --img 640 --batch 8 --conf 0.001 --iou 0.65 --device 0 --weights runs/train/yolov7/weights/best.pt --name yolov7_640_val --project test_result
# %%
