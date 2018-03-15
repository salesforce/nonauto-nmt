python run.py   --prefix [time] --gpu 2 
                --eval-every 500 \
                --dataset iwslt \
                --tensorboard \
                --level subword \
                --use_mask \
                --data_prefix ../data/ \
                --use_wo \
                --share_embeddings \
                --debug
                #--params base-t2t \
                #--fast --use_alignment --diag --positional_attention \
