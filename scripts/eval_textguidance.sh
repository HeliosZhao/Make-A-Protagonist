
## LR 3e-5, drop rate 0.2, 0.3, 0.5
NOISE=$1
for CFG in 12.5 10 7.5 5
do
    for TG in 1 2.5 5 7.5 10
    do
        python eval_unclip_tuneavideo.py --config configs/ikun/sd2/eval-ikun-768-8-long-drop-tem.yaml --options validation_data.noise_level=$NOISE validation_data.guidance_scale=$CFG validation_data.text_guidance_scale=$TG
    done
done
