python run.py --prefix [time] --gpu 0 \
            --fast \
            --eval-every 500 \
            --dataset wmt16-deen --load_vocab \
            --language deen \
            --tensorboard \
            --level subword \
            --use_mask \
            --diag --positional_attention \
            --use_wo --share_embeddings \
            --use_posterior_order --preordering --fertility_only \
            --seq_dist --remove_eos \
            --share_encoder --finetune_encoder \
	    --teacher 10.17_08.22.wmt16-ende_subword_512_512_6_8_0.100_16000___ \
            #--debug \
            # --load_from 10.20_23.28.wmt16-roen_subword_fast_512_512_6_8_0.100_16000___ \
            # --resume \
            

