python run.py --prefix [time] --gpu 3 --eval-every 2000 \
              --dataset wmt16-ende \
              --tensorboard \
              --level subword \
              --use_mask \
              --batchsize 3072 \
              --use_wo \
              --share_embeddings \
              --data_prefix ../data/ \
              --language deen \
              #--debug \
              #--load_dataset \
