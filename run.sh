source venv3.9/bin/activate
python train.py --model densenet201 --wandb --no-sampler --batch_size 600 --img_size 224 --device parallel --num_worker 16 --tags ["no-sampler"] --cache
python train.py --model densenet201 --wandb --sampler --batch_size 600 --img_size 224 --device parallel --num_worker 16 --tags ["with sampler"] --cache
python train.py --model densenet201 --wandb --sampler --batch_size 600 --img_size 224 --device parallel --num_worker 16 --tags ["sampler","augment"] --augment_prob 0.1 --augment_intensity 0.1 --cache
python train.py --model densenet201 --wandb --sampler --batch_size 600 --img_size 224 --device parallel --num_worker 16 --tags ["sampler","label_smoothing"] --label_smoothing 0.1 --cache
