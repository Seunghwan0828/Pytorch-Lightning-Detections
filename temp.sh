# train
# python faster.py  --train_path=data/images/images/train/ --train_annt_path=data/train_annotations.json \
# --weight_path=weights/faster/ --project=faster --max_epochs=50 --batch_size=16

# python ssd.py  --train_path=data/images/train/ --train_annt_path=data/train_annotations.json \
# --weight_path=weights/ssd/ --project=ssd --max_epochs=100 --batch_size=32


# eval
# python faster.py  --valid_path=data/images/images/valid/ --valid_annt_path=data/valid_annotations.json \
#--batch_size=16 --mode=test --checkpoint=weights/faster/best.ckpt

# python ssd.py  --valid_path=data/images/images/valid/ --valid_annt_path=data/valid_annotations.json \
# --batch_size=32 --mode=test --checkpoint=weights/ssd/best.ckpt

