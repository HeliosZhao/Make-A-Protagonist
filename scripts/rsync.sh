rsync -avz --exclude='.git' --exclude-from='Tune-A-Video/.gitignore' Tune-A-Video/ yyzhao@cvrp4:/data/yyzhao/Tune-A-Video/

rsync -avz --exclude='.git' --exclude-from='.gitignore' * yyzhao@cvrp4:/data/yyzhao/Tune-A-Video/

rsync -avz --exclude='.git' --exclude-from='.gitignore' * yyzhao@lab:/home/yyzhao/diffusion/Tune-A-Video/

## NOTE if specified the mixed_precision, in default, it use many gpus, to use single gpu, should use --num_processes=1 
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2

CUDA_VISIBLE_DEVICES=0 python eval_tuneavideo.py --config configs/ikun/eval-ikun-512.yaml --options output_dir=outputs/eval-ikun-seed33 seed=33

CUDA_VISIBLE_DEVICES=1 python eval_tuneavideo.py --config configs/ikun/eval-ikun-512.yaml --options output_dir=outputs/eval-ikun-seed34 seed=34