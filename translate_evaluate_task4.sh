#!/usr/bin/env bash

dest_dir=output_task_4
mkdir -p $dest_dir

for i in 0 1 2 3 4 5 6 7 8 9 10 11
do
  SECONDS=0
  echo "Performing translation with length normalization by gamma $i"
  python translate_beam.py --data data_asg4/prepared_data --checkpoint-path checkpoints_asg4/checkpoint_best.pt --output \
  $dest_dir/model_translation_"$i".txt --cuda True --beam-size 3 --alpha 0.1 --gamma $i

  # restore translation from BPE
  sed -r 's/(@@ )|(@@ ?$)//g' $dest_dir/model_translation_"$i".txt > $dest_dir/model_translation_"$i".out

  # post process for assignment 4 data
  cat $dest_dir/model_translation_"$i".out | perl moses_scripts/detruecase.perl | \
  perl moses_scripts/detokenizer.perl -q -l en > $dest_dir/translation_"$i".txt
done

echo "Translation and evaluation results saved in $dest_dir"