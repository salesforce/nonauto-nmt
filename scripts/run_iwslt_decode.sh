python run.py   --prefix [time] --gpu 5 \
                --eval-every 500 \
                --dataset iwslt --load_vocab \
                --tensorboard \
                --level subword \
                --load_from 09.23_00.04.iwslt_subword_278_507_5_2_0.079_746___ \
                --use_mask \
                --mode test \
                --beam_size 1 \
                --no_write \
                --test_set IWSLT16.TED.tst2014 \
                #--output_fer \
                #--params base-t2t \
                #--fast --use_alignment --diag --positional_attention \
