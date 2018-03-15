python run.py   --prefix [time] --gpu 5 \
                --dataset wmt16-ende --language ende --load_vocab \
                --level subword \
                --load_from 10.10_23.04.wmt16-ende_subword_512_512_6_8_0.100_16000___ \
                --use_mask --use_wo --share_embeddings \
                --mode test \
                --test_set newstest2014 \
                --beam_size 4 \
