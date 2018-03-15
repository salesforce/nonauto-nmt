python run.py   --prefix [time] --gpu 6 \
                --eval-every 500 --fast --tensorboard \
                --load_vocab \
                --diag --remove_eos \
                --positional_attention \
                --fertility \
                --teacher 09.23_00.04.iwslt_subword_278_507_5_2_0.079_746___ \
                --load_from 10.24_16.11.iwslt_subword_fast_278_507_5_2_0.079_746___ \
                --debug \
                --mode test \
                --decode_mode search \
                --batch_size 100 \
                --beam_size 5 \
                --f_size 20 \
                --rerank_by_bleu \
                --no_write \
                --old 
                #--distillation \
 
