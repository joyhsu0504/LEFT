# Left

## ReferIt3D Dataset (SR3D)

**Step 1**: Prepare the dataset. Our dataset download process follows the [ReferIt3D benchmark](https://github.com/referit3d/referit3d).

Specifically, you will need to
- (1) Download `sr3d_train.csv` and `sr3d_test.csv` from this [link](https://drive.google.com/drive/folders/1DS4uQq7fCmbJHeE-rEbO8G1-XatGEqNV)
- (2) Download scans from ScanNet and process them according to this [link](https://github.com/referit3d/referit3d/blob/eccv/referit3d/data/scannet/README.md). This should result in a `keep_all_points_with_global_scan_alignment.pkl` file.


**Step 2**: Install the necessary packages.


Install the referit3d python package from [ReferIt3D](https://github.com/referit3d/referit3d).
```bash
  git clone https://github.com/referit3d/referit3d
  cd referit3d
  pip install -e .
```

Compile CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413).
```bash
  cd left/nn/point_net_pp/pointnet2
  python setup.py install
```

**Step 3**: Train. Here, `$scannet` is the path to `keep_all_points_with_global_scan_alignment.pkl`, `$scannet_split_pre_fix` is the path prefix to `scannetv2_train.txt` and `scannetv2_val.txt`, `$referit` is the path to `sr3d_train.csv`. You can download the files for `$gt_idx_to_class` (which is used for classification supervision, as in NS3D) and `$train`, `$test` from this [link].

```bash
jac-run scripts/trainval-referit3d.py --desc experiments/desc_neuro_codex_referit3d.py \
  --scannet-file $scannet --scannet-split-pre-fix $scannet_split_pre_fix --referit3D-file $referit --gt-idx-to-class $gt_idx_to_class \
  --parsed-train-path $train --parsed-test-path $test \
  --validation-interval 10 --save-interval 10 --lr 0.0001 --epochs 5000

```

**Step 4**: Evaluate. You can find our trained checkpoint for `$load_path` from this [link].

```bash
jac-run scripts/trainval-referit3d.py --desc experiments/desc_neuro_codex_referit3d.py \
  --scannet-file $scannet --scannet-split-pre-fix $scannet_split_pre_fix --referit3D-file $referit --gt-idx-to-class $gt_idx_to_class \
  --parsed-train-path $train --parsed-test-path $test \
  --load $load_path --evaluate
```

