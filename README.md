# Automated Virtual Bronchoscopy Pipeline

This project builds an end-to-end airway navigation pipeline:

```text
CT volume
-> WingsNet lumen/wall inference
-> lumen skeletonization
-> airway graph
-> trachea-rooted navigation path
-> spline-smoothed centerline path
-> airway mesh + path visualization
-> VTK/PyVista flythrough rendering
-> graph/path validation metrics
```

The current pipeline supports AirRC-style processed CT tensors and raw LIDC DICOM CT series.

## Important Fixes Applied

Two fixes were needed before LIDC visualization became reasonable:

1. `infer_wingsnet_full_volume.py` now uses the final model output head for tuple/list outputs:

```python
output = output[-1]
```

This matches `evaluation.py` and avoids using an earlier deep-supervision head.

2. `skeletonize_airrc_case.py` now supports:

```bash
--root-mode timi-trachea
```

This uses TIMI-style branch parsing to pick the trachea branch instead of choosing a root by simple endpoint geometry.

## Main Scripts

```text
preprocess_lidc_for_inference.py
```

Reads raw LIDC DICOM CT, resamples to target spacing, clips HU to `[-1000, 400]`, normalizes to `[0, 1]`, and saves `.npy`.

```text
infer_wingsnet_full_volume.py
```

Loads `wingsnet_best.pth`, runs sliding-window inference, and saves lumen/wall probability volumes and masks.

```text
skeletonize_airrc_case.py
```

Thresholds predicted lumen, skeletonizes it, builds a graph, chooses a trachea root, and exports planned paths.

```text
smooth_airway_path.py
```

Converts raw graph-node paths into spline-smoothed centerline trajectories for virtual bronchoscopy camera motion. The raw paths remain saved for validation and traceability.

```text
visualizer_airway_mesh_path.py
```

Creates STL/OBJ mesh and PNG/GIF validation render with the planned path overlaid. It also supports visual-only mesh cleanup options such as largest component, hole filling, closing, and mesh smoothing.

```text
flythrough_airway_path.py
```

Creates an intraluminal virtual bronchoscopy GIF using the cleaned airway mesh and the lumen-projected smoothed path. This uses PyVista/VTK for rendering. The camera endpoint is the last point of `pred_lumen_smoothed_paths.json`, not the raw LIDC nodule coordinate, so the camera remains inside the predicted lumen.

```text
validate_centerline_graph.py
```

Computes graph/path validation metrics.

```text
extract_lidc_poi_target.py
```

Reads LIDC XML nodule annotations and converts nodule centroids into resampled CT voxel coordinates (`target_zyx`) for target-driven path planning.

```text
run_airway_pipeline.py
```

Runs inference, skeletonization/path planning, spline path smoothing, and visualization in one command.

## LIDC Preprocessing

For raw LIDC DICOM input, first preprocess the CT series.

Example DICOM folder:

```bash
CASE_ID=LIDC-IDRI-0001

CT_DIR=/home/opat90op/AMS_Project/datasets_new/lidc/lidc_idri/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/CT_1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192
```

Run:

```bash
python preprocess_lidc_for_inference.py \
  --dicom-dir ${CT_DIR} \
  --case-id ${CASE_ID} \
  --output-root /home/opat90op/AMS_Project/datasets_new/processed_lidc \
  --target-spacing 1,1,1 \
  --save-nifti
```

Outputs:

```text
/home/opat90op/AMS_Project/datasets_new/processed_lidc/images/LIDC-IDRI-0001_ct.npy
/home/opat90op/AMS_Project/datasets_new/processed_lidc/images/LIDC-IDRI-0001_ct.nii.gz
/home/opat90op/AMS_Project/datasets_new/processed_lidc/metadata/LIDC-IDRI-0001_meta.json
```

## Full Pipeline Run

Run the fixed pipeline on a processed AirRC CT:

```bash
CASE=1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860
DATA_ROOT=/home/opat90op/AMS_Project/datasets_new

python run_airway_pipeline.py \
  --ct ${DATA_ROOT}/processed_airrc/images/${CASE}_ct.npy \
  --case-id ${CASE} \
  --data-root ${DATA_ROOT} \
  --checkpoint saved_model/wingsnet_best.pth \
  --model-module WingsNet \
  --model-class WingsNet \
  --model-kwargs '{"in_channel": 1, "n_classes": 2}' \
  --device cpu \
  --mask-threshold 0.5 \
  --skeleton-threshold 0.2 \
  --prune-length 12 \
  --preserve-generations 6 \
  --root-mode timi-trachea \
  --trachea-root-end min-z \
  --target-mode farthest-endpoint \
  --spacing 1,1,1 \
  --make-video
```

Run the fixed pipeline on the preprocessed LIDC CT:

```bash
CASE_ID=LIDC-IDRI-0001_resampled_fixed
DATA_ROOT=/home/opat90op/AMS_Project/datasets_new

python run_airway_pipeline.py \
  --ct ${DATA_ROOT}/processed_lidc/images/LIDC-IDRI-0001_ct.npy \
  --case-id ${CASE_ID} \
  --data-root ${DATA_ROOT} \
  --checkpoint saved_model/wingsnet_best.pth \
  --model-module WingsNet \
  --model-class WingsNet \
  --model-kwargs '{"in_channel": 1, "n_classes": 2}' \
  --device cpu \
  --no-normalize \
  --mask-threshold 0.5 \
  --skeleton-threshold 0.2 \
  --prune-length 12 \
  --preserve-generations 6 \
  --root-mode timi-trachea \
  --trachea-root-end min-z \
  --target-zyx 224,258,222 \
  --spacing 1,1,1 \
  --make-video
```

If the trachea root appears to start at the wrong end, rerun with:

```bash
--trachea-root-end max-z
```

## Inference Only

Use this when you only need predicted lumen/wall volumes:

```bash
CASE=1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860
DATA_ROOT=/home/opat90op/AMS_Project/datasets_new

python infer_wingsnet_full_volume.py \
  --checkpoint saved_model/wingsnet_best.pth \
  --ct ${DATA_ROOT}/processed_airrc/images/${CASE}_ct.npy \
  --output-dir ${DATA_ROOT}/predictions/${CASE} \
  --model-module WingsNet \
  --model-class WingsNet \
  --model-kwargs '{"in_channel": 1, "n_classes": 2}' \
  --device cpu \
  --threshold 0.5 \
  --save-binary
```

For LIDC CT saved by `preprocess_lidc_for_inference.py`, add `--no-normalize`:

```bash
CASE_ID=LIDC-IDRI-0001_resampled_fixed
DATA_ROOT=/home/opat90op/AMS_Project/datasets_new

python infer_wingsnet_full_volume.py \
  --checkpoint saved_model/wingsnet_best.pth \
  --ct ${DATA_ROOT}/processed_lidc/images/LIDC-IDRI-0001_ct.npy \
  --output-dir ${DATA_ROOT}/predictions/${CASE_ID} \
  --model-module WingsNet \
  --model-class WingsNet \
  --model-kwargs '{"in_channel": 1, "n_classes": 2}' \
  --device cpu \
  --no-normalize \
  --threshold 0.5 \
  --save-binary
```

## Output Folders

For the example above:

```text
/home/opat90op/AMS_Project/datasets_new/predictions/LIDC-IDRI-0001_resampled_fixed/
/home/opat90op/AMS_Project/datasets_new/centerlines/LIDC-IDRI-0001_resampled_fixed_thr02/
/home/opat90op/AMS_Project/datasets_new/visualizations/LIDC-IDRI-0001_resampled_fixed_thr02/
```

Main outputs:

```text
pred_lumen.npy
pred_wall.npy
pred_lumen_mask.npy
pred_wall_mask.npy
pred_lumen_centerline.npy
pred_lumen_centerline_pruned.npy
pred_lumen_centerline_graph.json
pred_lumen_centerline_pruned_graph.json
pred_lumen_paths.json
pred_lumen_smoothed_paths.json
airway_lumen_mesh.stl
airway_lumen_mesh.obj
airway_mesh_path_overlay.png
airway_mesh_path_overlay.gif
```

`pred_lumen_paths.json` contains the raw shortest path through skeleton graph nodes. `pred_lumen_smoothed_paths.json` contains the centerline spline used for rendering and virtual bronchoscopy navigation.

To preserve a deeper skeleton tree during pruning, use:

```bash
--preserve-generations 6
```

This labels graph nodes by branchpoint depth from the selected trachea root and protects nodes through generation 6 from short-terminal-branch pruning. For LIDC nodule workflows, keep navigation target selection POI-based with `--target-zyx`.

## LIDC XML POI Targets

LIDC XML annotations can be used as nodule POIs. This is better than `--target-mode farthest-endpoint`, because the path becomes:

```text
trachea -> radiologist nodule POI / nearest airway access point
```

First find the XML matching the CT SeriesInstanceUID:

```bash
XML_ROOT=/home/opat90op/AMS_Project/datasets_new/lidc/lidc_idri/LIDC-XML-only/tcia-lidc-xml

grep -R "1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192" ${XML_ROOT}
```

For `LIDC-IDRI-0001`, the matching XML used was:

```text
/home/opat90op/AMS_Project/datasets_new/lidc/lidc_idri/LIDC-XML-only/tcia-lidc-xml/185/069.xml
```

Extract POI targets:

```bash
CASE_ID=LIDC-IDRI-0001

CT_DIR=/home/opat90op/AMS_Project/datasets_new/lidc/lidc_idri/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/CT_1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192

XML=/home/opat90op/AMS_Project/datasets_new/lidc/lidc_idri/LIDC-XML-only/tcia-lidc-xml/185/069.xml

python extract_lidc_poi_target.py \
  --xml ${XML} \
  --dicom-dir ${CT_DIR} \
  --metadata /home/opat90op/AMS_Project/datasets_new/processed_lidc/metadata/${CASE_ID}_meta.json \
  --output-json /home/opat90op/AMS_Project/datasets_new/processed_lidc/metadata/${CASE_ID}_poi_targets.json
```

Example targets printed for this case included:

```text
Nodule 001 target_zyx=[224, 258, 222]
Nodule 002 target_zyx=[190, 117, 131]
Nodule 003 target_zyx=[165, 191, 283]
Nodule 004 target_zyx=[165, 223, 276]
```

Repeated targets around `[224, 258, 222]` represent multiple reader annotations of the same main nodule cluster.

## POI-Targeted Path Planning

Use a POI target instead of the arbitrary farthest endpoint:

```bash
CASE_ID=LIDC-IDRI-0001_resampled_fixed
THR=0.2
TAG=thr02_poi_main

python skeletonize_airrc_case.py \
  --input /home/opat90op/AMS_Project/datasets_new/predictions/${CASE_ID}/pred_lumen.npy \
  --output-dir /home/opat90op/AMS_Project/datasets_new/centerlines/${CASE_ID}_${TAG} \
  --threshold ${THR} \
  --keep-largest \
  --prune-length 12 \
  --preserve-generations 6 \
  --root-mode timi-trachea \
  --trachea-root-end min-z \
  --target-zyx 224,258,222

python smooth_airway_path.py \
  --paths-json /home/opat90op/AMS_Project/datasets_new/centerlines/${CASE_ID}_${TAG}/pred_lumen_paths.json \
  --output-json /home/opat90op/AMS_Project/datasets_new/centerlines/${CASE_ID}_${TAG}/pred_lumen_smoothed_paths.json \
  --lumen-mask /home/opat90op/AMS_Project/datasets_new/predictions/${CASE_ID}/pred_lumen.npy \
  --threshold ${THR} \
  --project-to-lumen
```

Important: LIDC nodule POIs are not airway-centerline points. The planner maps the POI to the nearest graph node, so the final point is the closest airway access point to the nodule, not necessarily the nodule center itself.

For the final `LIDC-IDRI-0001` run, the clinical POI was:

```text
requested LIDC target zyx: [224, 258, 222]
selected lumen-safe airway endpoint zyx: [220, 249, 216]
target distance: 11.53 voxels
target generation: 8
```

This means the planned bronchoscopy path reaches the nearest connected predicted airway point toward the nodule. It does not force the camera outside the lumen to the nodule centroid.

## POI Visualization

Render the POI-targeted path:

```bash
python visualizer_airway_mesh_path.py \
  --mask /home/opat90op/AMS_Project/datasets_new/predictions/${CASE_ID}/pred_lumen.npy \
  --paths-json /home/opat90op/AMS_Project/datasets_new/centerlines/${CASE_ID}_${TAG}/pred_lumen_smoothed_paths.json \
  --output-dir /home/opat90op/AMS_Project/datasets_new/visualizations/${CASE_ID}_${TAG} \
  --threshold ${THR} \
  --tube-radius 0.8 \
  --endpoint-point-size 6 \
  --make-video
```

If the green/black endpoint dots look too large, reduce:

```bash
--endpoint-point-size 4
```

or hide endpoint dots:

```bash
--endpoint-point-size 0
```

## Visual Mesh Cleanup

The TIMI postprocessing script mainly keeps the largest connected component and fills holes. The visualizer now exposes similar visual-only cleanup options plus mesh smoothing.

Use this to make distal bronchi less fragmented in PNG/GIF/STL/OBJ output:

```bash
python visualizer_airway_mesh_path.py \
  --mask /home/opat90op/AMS_Project/datasets_new/predictions/${CASE_ID}/pred_lumen.npy \
  --paths-json /home/opat90op/AMS_Project/datasets_new/centerlines/${CASE_ID}_${TAG}/pred_lumen_smoothed_paths.json \
  --output-dir /home/opat90op/AMS_Project/datasets_new/visualizations/${CASE_ID}_${TAG}_smooth \
  --threshold ${THR} \
  --tube-radius 0.8 \
  --endpoint-point-size 6 \
  --mesh-keep-largest \
  --mesh-fill-holes \
  --mesh-closing-radius 1 \
  --mesh-smooth-iterations 15 \
  --make-video
```

These options affect only the rendered mesh. They do not change the predicted path or validation metrics.

If smoothing creates unrealistic connections, reduce the cleanup:

```bash
--mesh-closing-radius 0 --mesh-smooth-iterations 8
```

For the current best `LIDC-IDRI-0001` output, the preferred cleanup is conservative:

```bash
python visualizer_airway_mesh_path.py \
  --mask /home/opat90op/AMS_Project/datasets_new/predictions/LIDC-IDRI-0001_topology_clean_prune12/pred_lumen.npy \
  --paths-json /home/opat90op/AMS_Project/datasets_new/centerlines/LIDC-IDRI-0001_topology_clean_prune12_thr02/pred_lumen_smoothed_paths.json \
  --graph-json /home/opat90op/AMS_Project/datasets_new/centerlines/LIDC-IDRI-0001_topology_clean_prune12_thr02/pred_lumen_centerline_pruned_graph.json \
  --output-dir /home/opat90op/AMS_Project/datasets_new/visualizations/LIDC-IDRI-0001_topology_clean_prune12_thr02_meshclean \
  --threshold 0.2 \
  --low-threshold 0.08 \
  --spacing 1,1,1 \
  --tube-radius 1.5 \
  --endpoint-point-size 0 \
  --mesh-keep-largest \
  --mesh-smooth-iterations 20 \
  --mesh-smooth-relaxation 0.02 \
  --make-video
```

This removes disconnected rendered clutter while avoiding `--mesh-opening-radius 1`, which can erase real thin distal branches.

## Virtual Bronchoscopy Flythrough

The flythrough uses the cleaned mesh plus the smoothed, lumen-projected path:

```bash
python flythrough_airway_path.py \
  --mesh /home/opat90op/AMS_Project/datasets_new/visualizations/LIDC-IDRI-0001_topology_clean_prune12_thr02_meshclean/airway_lumen_mesh.stl \
  --paths-json /home/opat90op/AMS_Project/datasets_new/centerlines/LIDC-IDRI-0001_topology_clean_prune12_thr02/pred_lumen_smoothed_paths.json \
  --output /home/opat90op/AMS_Project/datasets_new/visualizations/LIDC-IDRI-0001_topology_clean_prune12_thr02_meshclean/airway_flythrough_bronchoscopy.gif \
  --spacing 1,1,1 \
  --frames 720 \
  --fps 24 \
  --lookahead 14 \
  --window-size 900,700 \
  --mesh-color "#c98270" \
  --mucosa-texture \
  --texture-strength 0.85 \
  --mesh-smooth-iterations 4 \
  --mesh-smooth-relaxation 0.005 \
  --view-angle 92 \
  --ambient 0.38 \
  --diffuse 0.75 \
  --specular 0.18 \
  --headlight-intensity 0.28
```

The GIF is qualitative. It is generated from a segmentation surface, so it cannot reproduce true optical bronchoscopy mucosa. The procedural texture and warm lighting are used only to make the view more bronchoscopy-like.

Primary flythrough outputs:

```text
airway_flythrough_bronchoscopy.gif
airway_flythrough_bronchoscopy.json
```

## Validation

For LIDC, do not pass `--target` unless a true airway lumen ground-truth mask exists. LIDC XML files are usually nodule annotations, not airway segmentations.

Run:

```bash
CASE_ID=LIDC-IDRI-0001_resampled_fixed
TAG=${CASE_ID}_thr02

python validate_centerline_graph.py \
  --pred-lumen /home/opat90op/AMS_Project/datasets_new/predictions/${CASE_ID}/pred_lumen.npy \
  --skeleton /home/opat90op/AMS_Project/datasets_new/centerlines/${TAG}/pred_lumen_centerline_pruned.npy \
  --graph-json /home/opat90op/AMS_Project/datasets_new/centerlines/${TAG}/pred_lumen_centerline_pruned_graph.json \
  --paths-json /home/opat90op/AMS_Project/datasets_new/centerlines/${TAG}/pred_lumen_paths.json \
  --threshold 0.2 \
  --short-branch-length 12 \
  --timi-tree-parse \
  --output-json /home/opat90op/AMS_Project/datasets_new/centerlines/${TAG}/centerline_validation_metrics.json
```

For the POI-targeted path, validate the POI output folder instead:

```bash
CASE_ID=LIDC-IDRI-0001_resampled_fixed
TAG=thr02_poi_main
THR=0.2

python validate_centerline_graph.py \
  --pred-lumen /home/opat90op/AMS_Project/datasets_new/predictions/${CASE_ID}/pred_lumen.npy \
  --skeleton /home/opat90op/AMS_Project/datasets_new/centerlines/${CASE_ID}_${TAG}/pred_lumen_centerline_pruned.npy \
  --graph-json /home/opat90op/AMS_Project/datasets_new/centerlines/${CASE_ID}_${TAG}/pred_lumen_centerline_pruned_graph.json \
  --paths-json /home/opat90op/AMS_Project/datasets_new/centerlines/${CASE_ID}_${TAG}/pred_lumen_paths.json \
  --threshold ${THR} \
  --short-branch-length 12 \
  --timi-tree-parse \
  --output-json /home/opat90op/AMS_Project/datasets_new/centerlines/${CASE_ID}_${TAG}/centerline_validation_metrics.json
```

Good signs:

```text
graph_component_count = 1
isolated_node_count = 0
path_inside_lumen_ratio close to 1.0
timi unreached branches low
timi multi-parent branches low
endpoint/branchpoint counts not exploding
```

## Notes On LIDC Generalization

The model was trained on AirRC airway labels, not LIDC airway labels. LIDC XML annotations do not provide full airway lumen/wall masks. Therefore, LIDC results should be treated as qualitative/structural validation unless airway ground truth is available.

For LIDC cases, the most important qualitative checks are:

```text
the mesh resembles a connected airway tree
the green root starts at the trachea
the red path stays inside the lumen
small noisy branches are limited
the path is not selected from a random distal endpoint
the flythrough endpoint stays inside the predicted lumen
```

## Recommended Next Improvements

1. Validate the final settings on multiple held-out AirRC and LIDC-like cases before making robustness claims.
2. Add stronger false-branch pruning using branch volume/radius/probability support.
3. Improve distal airway depth only when the lumen prediction contains connected distal airways.
4. Fine-tune or calibrate on LIDC-like airway labels if they become available.

## Distal/Topology-Aware Retraining

To improve distal bronchus recovery, regenerate AirRC patches with endpoint/thin-airway oversampling. The current patch extractor splits by case UID before patch extraction, avoiding train/validation leakage:

```bash
python extract_airrc_patches.py \
  --patches-per-case 32 \
  --distal-patch-fraction 0.5 \
  --boundary-patch-fraction 0.3 \
  --distal-radius-percentile 35
```

For cluster use:

```bash
sbatch extract_patches_gpu.sh
```

Then fine-tune from the current checkpoint with moderate distal weighting, augmentation, and topology/clDice-style supervision:

```bash
sbatch train_gpu.sh
```

The current best LIDC qualitative result used:

```text
saved_model_topology/wingsnet_best.pth
```

Do not automatically promote a checkpoint only because validation loss or distal recall improves. For LIDC nodule navigation, compare checkpoints using:

```text
target_distance_voxels
target_generation
endpoint/branchpoint counts
path_inside_lumen_ratio
visual false-branch/noise level
```

For `LIDC-IDRI-0001`, `saved_model_topology/wingsnet_best.pth` was preferred over `saved_model_topology/wingsnet_best_distal.pth`, because it reached closer to the target while keeping a cleaner trachea-rooted path.
