
## LR 3e-5, drop rate 0.2, 0.3, 0.5
for NOISE in 0 200 400 900 10001
do
    python eval_unclip_tuneavideo.py --config configs/ikun/sd2/eval-ikun-768-8-long-drop-tem.yaml --options validation_data.noise_level=$NOISE
done