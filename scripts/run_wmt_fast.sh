python run.py   --prefix [time] --gpu 1 \
                --eval-every 500 --fast --tensorboard \
                --dataset wmt16-ende --language ende \
                --load_vocab \
                --level subword \
                --use_mask --diag --positional_attention \
                --load_from 10.22_18.30.wmt16-ende_subword_fast_512_512_6_8_0.100_16000___ \
                --resume \
                --batchsize 1600 \
		            --preordering \
                --use_wo --share_embeddings \
                --use_posterior_order \
                --fertility_only \
		            --remove_eos \
                --seq_dist \
                --debug
                #--debug
                #--teacher 10.10_23.04.wmt16-ende_subword_512_512_6_8_0.100_16000___ \
                #--load_from 10.22_06.01.wmt16-ende_subword_fast_512_512_6_8_0.100_16000___ \
                # --share_encoder --finetune_encoder \
