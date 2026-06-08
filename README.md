# 4-view ADPT to 3D reconstruction for 3D-Primate-Tracker

This package adapts the provided MATLAB calibration converter (`untitled.m`) and DLT triangulation demo (`step4.py`) to reconstruct 3D macaque body keypoints from 4-view ADPT/DLC-style tracking CSVs.

## 1. Build multi-view projection matrices in MATLAB

Put all pairwise stereo calibration files in one folder, for example:

```text
calibrationSession_1.mat
calibrationSession_2.mat
calibrationSession_3.mat
calibrationSession_4.mat
```

Then run:

```matlab
build_projection_matrices_4view('D:/3D-Primate-Tracker/calibration', ...
                               'D:/3D-Primate-Tracker/calibration/projection_matrices.mat', ...
                               4, 1);
```

`main_cam_idx` uses MATLAB 1-based indexing. If ADPT files are named `camera-0`, `camera-1`, `camera-2`, `camera-3`, then they usually map to MATLAB camera indices `1`, `2`, `3`, `4`, and Python projection indices `0`, `1`, `2`, `3`.

## 2. Reconstruct 3D coordinates in Python

```bash
python reconstruction/reconstruct_3d_adpt_multiview.py \
  --projection-mat D:/3D-Primate-Tracker/calibration/projection_matrices.mat \
  --camera-csvs \
      D:/tracking/camera0.csv \
      D:/tracking/camera1.csv \
      D:/tracking/camera2.csv \
      D:/tracking/camera3.csv \
  --projection-indices 0 1 2 3 \
  --likelihood-threshold 0.5 \
  --min-views 2 \
  --output-csv D:/tracking/reconstruction_3d.csv
```

The output CSV uses the requested two-row header format:

```text
Nose,Nose,Nose,Left Eye,Left Eye,Left Eye,...
x,y,z,x,y,z,...
```

If two animals are present, output labels are automatically prefixed as `individual_0_Nose`, `individual_1_Nose`, etc. To reconstruct only one individual and keep plain body-part names:

```bash
python reconstruction/reconstruct_3d_adpt_multiview.py \
  --projection-mat projection_matrices.mat \
  --camera-csvs camera0.csv camera1.csv camera2.csv camera3.csv \
  --projection-indices 0 1 2 3 \
  --individual individual_0 \
  --output-csv individual_0_3d.csv
```

To add a frame column:

```bash
--include-frame
```

To export C3D as well:

```bash
--output-c3d output_3d.c3d --fps 30
```

C3D export requires:

```bash
pip install ezc3d
```

## 3. Run M³ behavioral manifold analysis

The package also includes the original M³ analysis scripts in `m3_analysis/`:

```text
m3_analysis/
├── m3_2D.py
├── m3_2D_single.py
├── m3_3D.py
└── README_m3_analysis.md
```

After reconstructing 4-view ADPT results into 3D keypoint CSV files, place the reconstructed files in the input directory expected by the M³ scripts or modify the `data_path` and `save_path` variables in each script.

Typical usage:

```bash
python m3_analysis/m3_2D_single.py
python m3_analysis/m3_2D.py
python m3_analysis/m3_3D.py
```

`m3_2D_single.py` is for solitary/single-animal analysis, `m3_2D.py` is for paired-animal 2D UMAP analysis, and `m3_3D.py` is for paired-animal 3D UMAP robustness/ablation analysis.
