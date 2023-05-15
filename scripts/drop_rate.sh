
## LR 3e-5, drop rate 0.2, 0.3, 0.5
for DROP in 0.2 0.3 0.5
do
    python train_unclip_tuneavideo.py --config configs/ikun/ikun-768-8-long-drop.yaml --options output_dir=outputs/ikun-768-8-long-drop$DROP train_data.image_embed_drop=$DROP
done