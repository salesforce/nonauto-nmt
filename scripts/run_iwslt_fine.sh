python run.py   --prefix [time] --gpu 2 \
                --eval-every 100 --fast --tensorboard \
                --level subword --load_vocab \
                --use_mask --diag --positional_attention \
                --use_alignment \
                --load_from 09.30_04.37.iwslt2_subword_fast_278_507_5_2_0.079_746___ \
                --teacher 09.23_00.04.iwslt_subword_278_507_5_2_0.079_746___ \
                --dataset iwslt \
                --disable_lr_schedule \
                --seq_dist \
                --beta1 0.5 \
                --beta2 0.5 \
                --word_dist \
		        --mode test \
                --batchsize 32000 \
                --no_write \
                --debug
                #--trainable_teacher \
                #--continue_teacher_training \
                #--debug \
                #--only_update_errors \
                #--critic \
                #--critic_only 1000000 \
                #--reverse_kl \
                #--critic_only 2000 \
                #--critic_only 2000 \
                #--entropy \
                #--trainable_teacher \
                #--batchsize 1024 \
                #--margin 0.0 \
                #--debug
                #--trainable_teacher \
                #--margin 1.4 \
                #--entropy \
                #--debug \
                #--teacher_discriminator --st \
                #--st \
                #--entropy \
                #--teacher_discriminator --st \
                # --debug \
                #--share_encoder --finetune_encoder \
                #--disable_lr_schedule \
                # --debug \
                # --no_source
                # --positional_attention \
                #--attention_discrimination --debug
                #--load_from 09.27_17.51.iwslt_subword_fast_278_507_5_2_0.079_746___ \
