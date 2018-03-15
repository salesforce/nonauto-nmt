python run.py   --prefix [time] --gpu 6 \
                --eval-every 100 --fast --tensorboard \
                --level subword --load_vocab \
                --use_mask --diag --positional_attention \
                --load_from 10.24_18.16.wmt16-deen_subword_fast_512_512_6_8_0.100_16000___ \
                --teacher 10.17_08.22.wmt16-ende_subword_512_512_6_8_0.100_16000___ \
                --dataset wmt16-deen --language deen \
                --disable_lr_schedule \
                --seq_dist \
                --beta1 0.75 \
                --beta2 0.5 \
                --word_dist \
                --use_wo --share_embeddings \
		            --preordering --use_posterior_order --fertility_only \
		            --remove_eos \
                --fertility_mode reinforce \
                --batchsize 1024 \
                #--debug \

