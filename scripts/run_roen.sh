python run.py --prefix [time] --gpu 5 \
              --eval-every 1000 \
              --dataset wmt16-roen \
              --language roen \
              --tensorboard \
              --level subword \
              --use_mask \
              --use_wo \
              --share_embeddings \
              --data_prefix ../data/ \
              --load_from 10.17_21.19.wmt16-enro_subword_512_512_6_8_0.100_16000___ \
              --mode test \
              --batchsize 32766 \
              #--debug
              #--mode decode \
              #--load_dataset \
