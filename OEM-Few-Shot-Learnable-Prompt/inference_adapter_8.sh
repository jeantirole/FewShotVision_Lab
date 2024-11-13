python inference_adapter.py --base-model-path /home/eric/srcs/FewShotSeg_Lab/FewShotVision_Lab/OEM-Few-Shot-Learnable-Prompt/outputs/base.pt \
--adapter-path /home/eric/srcs/FewShotSeg_Lab/FewShotVision_Lab/OEM-Few-Shot-Learnable-Prompt/logs/1731209861/weights/epoch04_loss1.6531_metric0.0855.pt \
--dataset-dir /home/eric/data/testset/8/images \
--class-idx 8 \
--outdir /home/eric/srcs/FewShotSeg_Lab/FewShotVision_Lab/OEM-Few-Shot-Learnable-Prompt/outputs/1110_2_class_8