#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstruct 3D macaque body keypoints from multi-view ADPT/DLC-style 2D tracking CSV files.

Input:
    - projection_matrices.mat containing cam_mat_all with shape (3, 4, n_cameras)
    - one ADPT/DLC-style CSV per camera, with 4 header rows:
      scorer / individual / bodyparts / coords, where coords include x, y, likelihood

Output:
    - two-row-header CSV formatted as:
      Nose,Nose,Nose,Left Eye,Left Eye,Left Eye,...
      x,y,z,x,y,z,...
      ...

Example:
    python reconstruct_3d_adpt_multiview.py \
        --projection-mat projection_matrices.mat \
        --camera-csvs camera0.csv camera1.csv camera2.csv camera3.csv \
        --projection-indices 0 1 2 3 \
        --likelihood-threshold 0.5 \
        --min-views 2 \
        --output-csv output_3d.csv
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io

Key = Tuple[str, str]  # (individual, bodypart)
TrackArrays = Dict[Key, Tuple[np.ndarray, np.ndarray, np.ndarray]]  # x, y, likelihood


def load_projection_matrices(mat_path: str, variable_name: str = "cam_mat_all") -> np.ndarray:
    """Load MATLAB projection matrices and return an array with shape (3, 4, n_cameras)."""
    mat = scipy.io.loadmat(mat_path)
    if variable_name not in mat:
        available = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"Variable '{variable_name}' not found in {mat_path}. Available variables: {available}")

    P = np.asarray(mat[variable_name], dtype=float)
    P = np.squeeze(P)

    if P.ndim != 3:
        raise ValueError(f"Projection matrix variable must be 3D, got shape {P.shape}")

    if P.shape[0] == 3 and P.shape[1] == 4:
        return P
    if P.shape[1] == 3 and P.shape[2] == 4:
        # Sometimes saved as (n_cameras, 3, 4)
        return np.transpose(P, (1, 2, 0))

    raise ValueError(
        f"Unsupported projection matrix shape {P.shape}. Expected (3, 4, n_cameras) "
        "or (n_cameras, 3, 4)."
    )


def _find_frame_column(columns: pd.MultiIndex):
    """Return the first column, which stores frame index in ADPT/DLC CSV exports."""
    return columns[0]


def load_adpt_csv(csv_path: str) -> Tuple[np.ndarray, TrackArrays, List[Key]]:
    """
    Load an ADPT/DLC-style tracking CSV.

    Returns:
        frames: frame indices from the first column
        tracks: dict mapping (individual, bodypart) -> (x, y, likelihood)
        key_order: bodypart order as it appears in this CSV
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2, 3])
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 4:
        raise ValueError(
            f"{csv_path} does not look like an ADPT/DLC-style CSV with 4 header rows."
        )

    frame_col = _find_frame_column(df.columns)
    frames = pd.to_numeric(df[frame_col], errors="coerce").to_numpy()

    # Build a column lookup: (individual, bodypart, coord) -> column
    lookup = {}
    key_order: List[Key] = []
    seen_keys = set()

    for col in df.columns[1:]:
        scorer, individual, bodypart, coord = col[:4]
        coord = str(coord).strip().lower()
        key = (str(individual), str(bodypart))
        if key not in seen_keys:
            seen_keys.add(key)
            key_order.append(key)
        lookup[(key[0], key[1], coord)] = col

    tracks: TrackArrays = OrderedDict()
    for individual, bodypart in key_order:
        x_col = lookup.get((individual, bodypart, "x"))
        y_col = lookup.get((individual, bodypart, "y"))
        l_col = lookup.get((individual, bodypart, "likelihood"))
        if x_col is None or y_col is None:
            continue
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
        if l_col is not None:
            likelihood = pd.to_numeric(df[l_col], errors="coerce").to_numpy(dtype=float)
        else:
            likelihood = np.ones_like(x, dtype=float)
        tracks[(individual, bodypart)] = (x, y, likelihood)

    return frames, tracks, list(tracks.keys())


def reconstruct_3d_point(points_2d: Sequence[Optional[Tuple[float, float]]],
                         projection_matrices: Sequence[np.ndarray]) -> np.ndarray:
    """
    Linear DLT triangulation from multiple camera views.

    Args:
        points_2d: list of (x, y) or None, one per selected camera view
        projection_matrices: list of 3x4 projection matrices aligned with points_2d

    Returns:
        3D point as (x, y, z). Returns NaNs if reconstruction is ill-conditioned.
    """
    A = []
    for point, P in zip(points_2d, projection_matrices):
        if point is None:
            continue
        x, y = point
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])

    if len(A) < 4:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    A = np.asarray(A, dtype=float)
    try:
        _, _, vt = np.linalg.svd(A)
        X_h = vt[-1, :]
        if np.isclose(X_h[3], 0.0):
            return np.array([np.nan, np.nan, np.nan], dtype=float)
        return X_h[:3] / X_h[3]
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan, np.nan], dtype=float)


def choose_keys(all_key_orders: Sequence[List[Key]],
                selected_individuals: Optional[Sequence[str]] = None) -> List[Key]:
    """Use the first camera's key order, optionally filtered by individual names."""
    if not all_key_orders:
        return []
    keys = all_key_orders[0]
    if selected_individuals:
        selected = set(selected_individuals)
        keys = [k for k in keys if k[0] in selected]
    return keys


def make_output_label(key: Key, n_individuals_in_output: int, keep_individual_prefix: bool) -> str:
    individual, bodypart = key
    if keep_individual_prefix or n_individuals_in_output > 1:
        return f"{individual}_{bodypart}"
    return bodypart


def reconstruct_multiview_csv(camera_csvs: Sequence[str],
                              projection_mat: str,
                              output_csv: str,
                              projection_indices: Optional[Sequence[int]] = None,
                              likelihood_threshold: float = 0.5,
                              min_views: int = 2,
                              max_frames: Optional[int] = None,
                              selected_individuals: Optional[Sequence[str]] = None,
                              keep_individual_prefix: bool = False,
                              include_frame: bool = False,
                              projection_variable: str = "cam_mat_all") -> pd.DataFrame:
    """Main reconstruction function."""
    if len(camera_csvs) < 2:
        raise ValueError("At least two camera CSV files are required for triangulation.")

    P_all = load_projection_matrices(projection_mat, projection_variable)
    n_available_cameras = P_all.shape[2]

    if projection_indices is None:
        projection_indices = list(range(len(camera_csvs)))
    projection_indices = list(projection_indices)

    if len(projection_indices) != len(camera_csvs):
        raise ValueError("--projection-indices must have the same length as --camera-csvs.")
    if any(i < 0 or i >= n_available_cameras for i in projection_indices):
        raise IndexError(
            f"Projection index out of range. Got {projection_indices}, "
            f"but projection_matrices contains {n_available_cameras} cameras."
        )

    frames_list = []
    tracks_by_camera: List[TrackArrays] = []
    key_orders = []

    for csv_path in camera_csvs:
        frames, tracks, key_order = load_adpt_csv(csv_path)
        frames_list.append(frames)
        tracks_by_camera.append(tracks)
        key_orders.append(key_order)

    n_frames = min(len(f) for f in frames_list)
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
    if n_frames <= 0:
        raise ValueError("No frames available for reconstruction.")

    keys = choose_keys(key_orders, selected_individuals)
    if not keys:
        raise ValueError("No bodypart keys found. Check CSV headers or --individual filters.")

    n_output_individuals = len({k[0] for k in keys})
    labels = [make_output_label(k, n_output_individuals, keep_individual_prefix) for k in keys]
    result = np.full((n_frames, len(keys) * 3), np.nan, dtype=float)

    selected_P = [P_all[:, :, i] for i in projection_indices]

    for frame_idx in range(n_frames):
        for key_idx, key in enumerate(keys):
            points_2d: List[Optional[Tuple[float, float]]] = []
            matrices_for_views: List[np.ndarray] = []

            for cam_idx, tracks in enumerate(tracks_by_camera):
                if key not in tracks:
                    continue
                x_arr, y_arr, likelihood_arr = tracks[key]
                x = x_arr[frame_idx]
                y = y_arr[frame_idx]
                conf = likelihood_arr[frame_idx]

                if (
                    np.isfinite(x) and np.isfinite(y) and np.isfinite(conf)
                    and conf >= likelihood_threshold
                ):
                    points_2d.append((float(x), float(y)))
                    matrices_for_views.append(selected_P[cam_idx])

            if len(points_2d) >= min_views:
                xyz = reconstruct_3d_point(points_2d, matrices_for_views)
                result[frame_idx, key_idx * 3:key_idx * 3 + 3] = xyz

        if (frame_idx + 1) % 1000 == 0:
            print(f"Processed {frame_idx + 1}/{n_frames} frames")

    # Build the requested two-line header: bodypart/bodypart/bodypart + x/y/z.
    top_header: List[str] = []
    coord_header: List[str] = []
    for label in labels:
        top_header.extend([label, label, label])
        coord_header.extend(["x", "y", "z"])

    out_df = pd.DataFrame(result, columns=pd.MultiIndex.from_arrays([top_header, coord_header]))
    if include_frame:
        # Insert a two-level frame column. The first camera's frame index is used.
        out_df.insert(0, ("frame", "frame"), frames_list[0][:n_frames])

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"3D reconstruction saved to: {output_csv}")
    return out_df


def save_c3d_from_reconstruction(output_csv: str, output_c3d: str, fps: float = 30.0) -> None:
    """Optional C3D export. Requires ezc3d: pip install ezc3d."""
    try:
        import ezc3d  # type: ignore
    except ImportError as exc:
        raise ImportError("C3D export requires ezc3d. Install it with: pip install ezc3d") from exc

    df = pd.read_csv(output_csv, header=[0, 1])
    if df.columns[0][0] == "frame":
        df_points = df.iloc[:, 1:]
    else:
        df_points = df

    values = df_points.to_numpy(dtype=float)
    n_frames = values.shape[0]
    n_points = values.shape[1] // 3
    labels = [df_points.columns[i * 3][0] for i in range(n_points)]

    points = np.full((4, n_points, n_frames), np.nan, dtype=float)
    reshaped = values.reshape(n_frames, n_points, 3)
    points[0:3, :, :] = np.transpose(reshaped, (2, 1, 0))
    points[3, :, :] = 1.0

    c3d = ezc3d.c3d()
    c3d["parameters"]["POINT"]["RATE"]["value"] = [fps]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = tuple(labels)
    c3d["data"]["points"] = points
    c3d.write(output_c3d)
    print(f"C3D saved to: {output_c3d}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D coordinates from 4-view ADPT/DLC-style 2D tracking CSV files."
    )
    parser.add_argument("--projection-mat", required=True, help="Path to projection_matrices.mat")
    parser.add_argument("--projection-variable", default="cam_mat_all", help="MAT variable name; default: cam_mat_all")
    parser.add_argument("--camera-csvs", nargs="+", required=True, help="ADPT CSV files ordered by camera")
    parser.add_argument(
        "--projection-indices", nargs="+", type=int, default=None,
        help="0-based indices into cam_mat_all for each CSV. Example for camera-0..3: 0 1 2 3"
    )
    parser.add_argument("--output-csv", required=True, help="Output 3D CSV path")
    parser.add_argument("--likelihood-threshold", type=float, default=0.5, help="Minimum 2D likelihood")
    parser.add_argument("--min-views", type=int, default=2, help="Minimum valid views required for triangulation")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum number of frames to process")
    parser.add_argument(
        "--individual", nargs="*", default=None,
        help="Optional individual names to reconstruct, e.g. individual_0 individual_1"
    )
    parser.add_argument(
        "--keep-individual-prefix", action="store_true",
        help="Always prefix output labels with individual name"
    )
    parser.add_argument(
        "--include-frame", action="store_true",
        help="Include frame index as the first output column"
    )
    parser.add_argument("--output-c3d", default=None, help="Optional output C3D path")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate for optional C3D export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reconstruct_multiview_csv(
        camera_csvs=args.camera_csvs,
        projection_mat=args.projection_mat,
        output_csv=args.output_csv,
        projection_indices=args.projection_indices,
        likelihood_threshold=args.likelihood_threshold,
        min_views=args.min_views,
        max_frames=args.max_frames,
        selected_individuals=args.individual,
        keep_individual_prefix=args.keep_individual_prefix,
        include_frame=args.include_frame,
        projection_variable=args.projection_variable,
    )
    if args.output_c3d:
        save_c3d_from_reconstruction(args.output_csv, args.output_c3d, args.fps)


if __name__ == "__main__":
    main()
