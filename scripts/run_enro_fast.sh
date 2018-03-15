python run.py --prefix [time] --gpu 1 \
            --fast \
            --eval-every 500 \
            --dataset wmt16-enro --load_vocab \
            --language enro \
            --tensorboard \
            --level subword \
            --use_mask \
            --diag --positional_attention \
            --use_wo --share_embeddings \
            --use_posterior_order --preordering --fertility_only \
            --load_from 10.20_23.32.wmt16-enro_subword_fast_512_512_6_8_0.100_16000___ \
            --resume \
            --seq_dist --remove_eos \
            #--debug
            #--mode decode \
            #--load_dataset \
            #--share_encoder --finetune_encoder \
            #--teacher 10.17_22.27.wmt16-enro_subword_512_512_6_8_0.100_16000___ \
