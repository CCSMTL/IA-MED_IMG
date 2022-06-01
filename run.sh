source venv3.9/bin/activate
python train.py --model densenet201 --wandb --no-sampler --batch_size 350 --img_size 224 --device parallel --num-worker 16 --tags ["no-sampler"]
python train.py --model densenet201 --wandb --sampler --batch_size 350 --img_size 224 --device parallel --num-worker 16 -- tags ["with sampler"]
