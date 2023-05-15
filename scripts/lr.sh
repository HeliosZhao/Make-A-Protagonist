
## LR 5e-5, 1e-4, 3e-4 drop rate 0.1
for LR in 5e-5 1e-4 3e-4
do
    python train_unclip_tuneavideo.py --config configs/ikun/ikun-768-8-long-drop.yaml --options output_dir=outputs/ikun-768-8-long-drop01-lr$LR learning_rate=$LR
done