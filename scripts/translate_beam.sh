#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..

lang=en-fr
trg=en

mkdir -p $base/translations
translations=$base/translations

echo "vary beam size up to k=25"
echo "************************************************************************************"
#for (( K=1; K<=25; K+=3 )); do
#    python translate_beam.py --data data/$lang/prepared --dicts data/$lang/prepared \
#                        --checkpoint-path assignments/03/baseline/checkpoints/checkpoint_last.pt \
#                        --output $translations/translation.beam.$K.$trg --batch-size 25 \
#                        --max-len 500 --alpha 0.0 --beam-size $K 
#done

#echo "detokenize translations"
#echo "************************************************************************************"
#for f in $translations/*.$trg; do
#    cat $f | perl moses_scripts/detruecase.perl | perl moses_scripts/detokenizer.perl -q -l $trg > $f.detok
#    rm $f
#done

#echo "calculate BLEU"
#echo "************************************************************************************"
#for f in $translations/*.detok; do
#    sacrebleu data/$lang/raw/test.$trg -i $f
#done

echo "translate with squared regularizer"
echo "************************************************************************************"
python translate_beam.py --data data/$lang/prepared --dicts data/$lang/prepared \
                            --checkpoint-path assignments/03/baseline/checkpoints/checkpoint_last.pt \
                            --output $translations/translation.regularized.txt --batch-size 25 \
                            --max-len 500 --beam-size 5 --regularizer True --lambda_ 0.5

echo "detokenize regularized translation"
echo "************************************************************************************"
cat $translations/translation.regularized.txt | perl moses_scripts/detruecase.perl | perl moses_scripts/detokenizer.perl -q -l $trg > $base/translations/translation.regularized.detok

echo "calculate BLEU for regularized translation"
echo "************************************************************************************"
sacrebleu data/$lang/raw/test.$trg -i $translations/translation.regularized.detok