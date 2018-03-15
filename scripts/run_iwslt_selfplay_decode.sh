python run.py   --prefix [time] --gpu 6 \
                --eval-every 500 --fast --tensorboard \
                --load_vocab \
                --diag --remove_eos \
                --positional_attention \
                --fertility \
                --load_from 11.14_02.27.iwslt_subword_fast_278_507_5_2_0.079_746___ \
                --mode self_play_decode \
                --decode_mode search \
                --batch_size 250 \
                --beam_size 5 \
                --f_size 10 \
                --rerank_by_bleu \
                --decode_every 500 \
                --train_every 500 \
                --old
                #--teacher 09.23_00.04.iwslt_subword_278_507_5_2_0.079_746___ \
                #--no_write \
                #--distillation \
                #--load_from 10.24_16.11.iwslt_subword_fast_278_507_5_2_0.079_746___ \

