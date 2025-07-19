torchrun main.py --warmup-epochs 5 --model MALA_S  --data-path /your/path/to/imagenet  --num_workers 16  --batch-size 128  --drop-path 0.05  --epoch 300 --dist-eval  --output_dir /ckpt
