#!/bin/bash
# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

# suffix of source language files
SRC=en

# suffix of target language files
TRG=de

# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=37000
tools=../tools
data=../data/iwslt/en-de

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=$tools/mosesdecoder

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=$tools/subword-nmt

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=$tools/nematus

# path to wmt16 scripts
wmt16=$tools/wmt16-scripts

# path to fastalign
fastalign=./fast_align

# convert the data into fast_align format --->
paste \
    $data/train.tags.en-de.bpe.dev.en2 \
    $data/train.tags.en-de.bpe.dev.de2 \
    | sed "s/$(printf '\t')/ ||| /g" > $data/train.tags.en-de.bpe.dev.fastalign2
l1=$(wc -l < "$data/train.tags.en-de.bpe.dev.fastalign2")

paste \
    $data/train.tags.en-de.bpe.en2 \
    $data/train.tags.en-de.bpe.de2 \
    | sed "s/$(printf '\t')/ ||| /g" > $data/train.tags.en-de.bpe.fastalign2
l2=$(wc -l < "$data/train.tags.en-de.bpe.fastalign2")

paste \
    $data/train.tags.en-de.bpe.dev.en2 \
    $data/train.tags.en-de.bpe.dev.decoded2 \
    | sed "s/$(printf '\t')/ ||| /g" > $data/train.tags.en-de.bpe.decode.dev.fastalign2
l3=$(wc -l < "$data/train.tags.en-de.bpe.decode.dev.fastalign2")

paste \
    $data/train.tags.en-de.bpe.en2 \
    $data/train.tags.en-de.bpe.decoded2 \
    | sed "s/$(printf '\t')/ ||| /g" > $data/train.tags.en-de.bpe.decode.fastalign2
l4=$(wc -l < "$data/train.tags.en-de.bpe.decode.fastalign2")

l2=$(($l2 + $l1))
l3=$(($l3 + $l2))
l4=$(($l4 + $l3))


echo $l1 $l2 $l3 $l4
cat $data/train.tags.en-de.bpe.dev.fastalign2 \
   $data/train.tags.en-de.bpe.fastalign2 \
   $data/train.tags.en-de.bpe.decode.dev.fastalign2 \
   $data/train.tags.en-de.bpe.decode.fastalign2 > $data/full_data.fastalign2

$fastalign \
   -i $data/full_data.fastalign2 \
   -v -p $data/full_data.fastlign2.cond \
   -N -d -o \
   > $data/full_data.fastalign2.aligned

sort -k1,1 -k3,3gr $data/full_data.fastlign2.cond \
     | sort -k1,1 -u \
       > $data/full_data.fastlign2.dict
echo $l1, $l11p
sed -n '1, 993p' $data/full_data.fastalign2.aligned > $data/train.tags.en-de.bpe.dev.aligned
sed -n '994, 209382p' $data/full_data.fastalign2.aligned > $data/train.tags.en-de.bpe.aligned
sed -n '209383, 210375p' $data/full_data.fastalign2.aligned > $data/train.tags.en-de.bpe.dev.decode.aligned
sed -n '210376, 418764p' $data/full_data.fastalign2.aligned > $data/train.tags.en-de.bpe.decode.aligned
echo 'done'
