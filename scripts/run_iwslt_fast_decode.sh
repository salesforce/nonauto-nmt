python run.py   --prefix [time] --gpu 5 \
                --eval-every 500 --fast --tensorboard \
                --level subword --load_vocab \
                --use_mask --diag \
        	      --preordering --use_posterior_order --fertility_only --remove_eos \
                --teacher  09.23_00.04.iwslt_subword_278_507_5_2_0.079_746___ \
                --load_from 10.24_16.11.iwslt_subword_fast_278_507_5_2_0.079_746___ \
                --positional_attention \
                --mode test_noisy \
                --beam_size 10 \
                --alpha 1 \
                --batchsize 100 \
                --old \
                --no_write \
                --fertility_mode mean \
                #--highway \
                #--mix_of_experts \
                #--debug
                #--debug
                #--orderless \
                #--debug \
