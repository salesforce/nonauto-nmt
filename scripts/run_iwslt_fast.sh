python run.py   --prefix [time] --gpu 5 \
                --eval-every 500 --fast --tensorboard \
                --level subword --load_vocab \
                --use_mask --diag \
        	    --preordering --use_posterior_order --fertility_only --remove_eos \
                --seq_dist \
                --teacher 09.23_00.04.iwslt_subword_278_507_5_2_0.079_746___ \
                --load_from 10.22_05.50.iwslt_subword_fast_278_507_5_2_0.079_746___ \
                --resume \
                --positional_attention \
                --debug \
                --mode test \
                --batchsize 1\
                --no_write \
                #--highway \
                #--mix_of_experts \
                #--debug
                #--debug
                #--orderless \
                #--debug \
