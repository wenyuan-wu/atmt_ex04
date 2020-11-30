#!/usr/bin/env bash

dest_dir=output_task_3
mkdir -p $dest_dir

for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
  SECONDS=0
  echo "Performing translation with length normalization by alpha $i"
  python translate_beam.py --data data_asg4/prepared_data --checkpoint-path checkpoints_asg4/checkpoint_best.pt --output \
  $dest_dir/model_translation_"$i".txt --cuda True --beam-size 3 --alpha $i

  # restore translation from BPE
  sed -r 's/(@@ )|(@@ ?$)//g' $dest_dir/model_translation_"$i".txt > $dest_dir/model_translation_"$i".out

  # post process for assignment 4 data
  cat $dest_dir/model_translation_"$i".out | perl moses_scripts/detruecase.perl | \
  perl moses_scripts/detokenizer.perl -q -l en > $dest_dir/translation_"$i".txt

  cat $dest_dir/translation_"$i".txt | sacrebleu data_asg4/raw_data/test.en > $dest_dir/result_"$i".txt
  echo "Time taken: $SECONDS seconds" >> $dest_dir/result_"$i".txt
done

echo "Translation and evaluation results saved in $dest_dir"