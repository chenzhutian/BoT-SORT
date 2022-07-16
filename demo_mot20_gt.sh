python3 tools/demo.py video --path videos/22_,.mp4 \
    -f yolox/exps/example/mot/yolox_x_mix_det.py \
    -c pretrained/bytetrack_x_mot20.tar \
    --fast-reid-config fast_reid/configs/MOT20/sbs_S50.yml \
    --fast-reid-weights pretrained/mot20_sbs_S50.pth \
    -g labels/bboxes/22_,.json \
    --out ./output \
    --with-reid --fp16 --fuse --save_result
