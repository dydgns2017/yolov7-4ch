# Ipykernel
# %%
!python create_cfg_4ch.py ## create sample dataset and cfg file
# %%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
!python train.py --workers 0 --device 0 --batch-size 4 --data test_dataset/data.yaml --img 640 640 --cfg yolov7-4ch.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml --epoch 5
# %%
!python test.py --data test_dataset/data.yaml --img 1280 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights runs/train/yolov73/weights/best.pt --name yolov7_1280_val --project test_result
# %%
