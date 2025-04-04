# python3 train_app/scripts/train.py train --data data/u-net.yml --project UNet &&
# python3 train_app/scripts/train.py train --data data/attentionu-net.yml --project AttUNet &&
# python3 train_app/scripts/train.py train --data data/mccm-net.yml --project MCCMNet

python3 train_app/scripts/train.py train --data submodule_configs/DEFAULT.yml --project DEFAULT && # approximately 30 hrs
python3 train_app/scripts/train.py train --data submodule_configs/NO_PIGM.yml --project NO_PIGM && # approximately 16 hrs
python3 train_app/scripts/train.py train --data submodule_configs/NO_MBDC.yml --project NO_MBDC && # approximately 9 hrs
python3 train_app/scripts/train.py train --data submodule_configs/NO_UNCERTAINTY.yml --project NO_UNCERTAINTY && # approximately 16 hrs
python3 train_app/scripts/train.py train --data submodule_configs/NO_FUSION.yml --project NO_FUSION && # approximately 16 hrs
python3 train_app/scripts/train.py train --data submodule_configs/DOUBLE_UNC.yml --project DOUBLE_UNC && # approximately 18 hrs
python3 train_app/scripts/train.py train --data submodule_configs/SINGLE_UNC.yml --project SINGLE_UNC # approximately 18 hrs
