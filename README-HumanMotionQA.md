# Left

## HumanMotionQA Dataset

**Step 1**: Prepare the dataset. Our dataset download process follows the [HumanMotionQA benchmark](https://github.com/markendo/HumanMotionQA/tree/master/BABEL-QA).


**Step 2**: Train.
```bash
jac-run scripts/trainval-humanmotion.py --desc experiments/desc_neuro_codex_humanmotion.py \
  --datadir $datadir --data-split-file $data_split_file --output-vocab-path $output_vocab_path --datasource humanml3d --no_gt_segments --temporal_operator conv1d \
  --parsed-train-path $train --parsed-test-path $test \
  --validation-interval 1 --save-interval 1 --lr 0.0005 --epochs 5000 --batch-size 4

```
Here, `$datadir` is the path to `BABEL-QA`, `$data_split_file` is the path to `split_question_ids.json`. You can download the files for `$output_vocab_path` and `$train`, `$test` from this download [link](https://downloads.cs.stanford.edu/viscam/LEFT/HumanMotionQA.zip).


**Step 3**: Evaluate.

```bash
jac-run scripts/trainval-humanmotion.py --desc experiments/desc_neuro_codex_humanmotion.py \
  --datadir $datadir --data-split-file $data_split_file --output-vocab-path $output_vocab_path --datasource humanml3d --no_gt_segments --temporal_operator conv1d \
  --parsed-train-path $train --parsed-test-path $test \
  --batch-size 4 --load $load_path --evaluate
```
You can find our trained checkpoint for `$load_path` from this download [link](https://downloads.cs.stanford.edu/viscam/LEFT/HumanMotionQA.zip).
