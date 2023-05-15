
## LR 3e-5, drop rate 0.2, 0.3, 0.5
for NOISE in 1.0 0.8 0.5 0.3 0.1
do
    python eval_unclip_controlavideo_prior.py --config configs/ikun/sd2/eval-ikun-768-8-unclip-prior.yaml --options validation_data.interpolate_embed_weight=$NOISE
done