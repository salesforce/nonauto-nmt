python run.py --prefix [time] --gpu 5 \
            --fast \
            --eval-every 500 \
            --dataset wmt16-roen --load_vocab \
            --language roen \
            --tensorboard \
            --level subword \
            --use_mask \
            --diag --positional_attention \
            --use_wo --share_embeddings \
            --use_posterior_order --preordering --fertility_only \
            --resume \
            --load_from 10.22_01.24.wmt16-roen_subword_fast_512_512_6_8_0.100_16000___ \
            --seq_dist --remove_eos \
            #--debug
            #--load_from 10.20_23.28.wmt16-roen_subword_fast_512_512_6_8_0.100_16000___ \
            #--share_encoder --finetune_encoder \
            #--teacher 10.17_21.19.wmt16-enro_subword_512_512_6_8_0.100_16000___ \

