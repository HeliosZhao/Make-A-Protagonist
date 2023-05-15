
VIDNAME=$1
PY_ARGS=${@:2}
echo $VIDNAME
echo $PY_ARGS

## extract images
python tools/vid2img.py -vr data/$VIDNAME.mp4 $PY_ARGS
## estimate pose
python tools/extract_conditions.py -o $VIDNAME -c depth
## estimate sketch
python tools/extract_conditions.py -o $VIDNAME -c sketch
## estimate pose
python tools/extract_conditions.py -o $VIDNAME -c openpose