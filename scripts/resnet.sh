LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" \
python main.py --config="./scripts/configs/cifar10/resnet.yaml" | tee log.txt 