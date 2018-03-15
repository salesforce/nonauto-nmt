python run.py   --prefix [time] --gpu 7 \
                --eval-every 500 --fast --tensorboard \
                --load_vocab \
                --diag --remove_eos \
                --positional_attention \
                --fertility \
                --load_from 11.14_02.27.iwslt_subword_fast_278_507_5_2_0.079_746___ \
                --mode self_play_train \
                --decode_mode argmax \
                --train_every 500 \
                --distillation \
                --disable_lr_schedule \
                --old
                #--teacher 09.23_00.04.iwslt_subword_278_507_5_2_0.079_746___ \
                #--no_write \
                #--distillation \
                #--load_from 10.24_16.11.iwslt_subword_fast_278_507_5_2_0.079_746___ \

