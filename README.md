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

## Start Here

This section is the complete setup and first-run path for a new user. Later sections document individual stages, training, validation, mesh export, and research interpretation.

### 1. Supported Environment

The main pipeline is intended for Linux or a Linux-based Slurm cluster.

Required:

- Python 3.10 or newer.
- PyTorch 2.x.
- The Python packages in `requirements.txt`.
- A trained WingsNet checkpoint.
- A CT DICOM series and, for POI navigation, its matching LIDC XML annotation.

Recommended for practical full-volume inference:

- An NVIDIA CUDA GPU with at least 16 GB memory.
- 16-24 GB system memory.
- Xvfb on headless Linux nodes for PyVista rendering.

Blender is not required for segmentation, path planning, validation, or flythrough rendering. It is only required by `blender_make_hollow_airway_shell.py` when producing a hollow physical-printing shell.

### 2. Create the Python Environment

From the repository root:

```bash
python3.10 -m venv idc_env
source idc_env/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install the PyTorch build appropriate for the machine first. On a managed cluster, use the CUDA/PyTorch command recommended by the administrator. For a CPU-only installation:

```bash
python -m pip install torch
```

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` deliberately does not install PyTorch because replacing a cluster-specific CUDA wheel with a generic wheel can disable GPU access.

Verify PyTorch before submitting a GPU job:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('torch cuda build:', torch.version.cuda)"
```

On a cluster, run this check inside an allocated GPU node. CUDA commonly reports `False` on a login node even when the environment is correct.

### 3. Configure Project Data

The default data location is:

```text
~/AMS_Project/datasets_new
```

To use another location, set one environment variable before running Python or submitting Slurm jobs:

```bash
export AVB_DATA_ROOT=/path/to/datasets_new
```

Create the expected base directories:

```bash
mkdir -p "$AVB_DATA_ROOT"/{airrc,airrc_patches,centerlines,final_validation,lidc,predictions,processed_airrc,processed_lidc,visualizations}
```

If `AVB_DATA_ROOT` is not set, replace it in this command with `$HOME/AMS_Project/datasets_new`.

The pipeline creates case-specific prediction, centerline, and visualization directories automatically.

### 4. Install the Required Checkpoint

The final checkpoint expected by the documented LIDC workflow is:

```text
saved_model_topology/wingsnet_best.pth
```

Place it relative to the repository root:

```bash
mkdir -p saved_model_topology
ls -lh saved_model_topology/wingsnet_best.pth
```

The `.pth` file is a required project artifact. If it is not distributed with the repository, obtain it from the project maintainer before running inference. Training data and source code alone do not recreate this exact checkpoint without retraining.

The original comparison checkpoint, used by the ablation suite, is:

```text
saved_model/wingsnet_best.pth
```

### 5. Verify the Installation

Run the automated setup check:

```bash
python check_setup.py \
  --checkpoint saved_model_topology/wingsnet_best.pth \
  --data-root "$AVB_DATA_ROOT"
```

For a GPU node:

```bash
python check_setup.py \
  --checkpoint saved_model_topology/wingsnet_best.pth \
  --data-root "$AVB_DATA_ROOT" \
  --require-cuda
```

Run the focused tests:

```bash
python -m unittest tests/test_core_utils.py
```

Do not proceed until the setup check finds the checkpoint and all required dependencies.

### 6. Download One LIDC Test Series

The `idc` command is provided by the `idc-index` package included in `requirements.txt`. A specific CT series can be downloaded directly by `SeriesInstanceUID`:

```bash
SERIES_UID=1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192

idc download "$SERIES_UID" \
  --download-dir "$AVB_DATA_ROOT/lidc/idc_downloads"
```

The LIDC XML annotation collection must also be available. Download `LIDC-XML-only.zip` from the official TCIA LIDC-IDRI collection page and extract it under:

```text
$AVB_DATA_ROOT/lidc/lidc_idri/LIDC-XML-only/
```

Official resources:

- IDC command-line download documentation: <https://github.com/ImagingDataCommons/idc-index>
- TCIA LIDC-IDRI collection and XML annotations: <https://www.cancerimagingarchive.net/collection/lidc-idri/>

The workflow verifies that the XML `SeriesInstanceUID` matches the selected DICOM series. It stops rather than silently converting coordinates from the wrong scan.

### 7. Run the Complete LIDC Workflow

On Slurm, one submission performs preprocessing, inference, graph extraction, POI ranking, target selection, path smoothing, target auditing, visualization, and optional flythrough rendering:

```bash
sbatch run_lidc_case_gpu.sh \
  --case-id LIDC-IDRI-0011 \
  --series-uid "$SERIES_UID" \
  --dicom-root "$AVB_DATA_ROOT/lidc/idc_downloads" \
  --xml-root "$AVB_DATA_ROOT/lidc/lidc_idri/LIDC-XML-only" \
  --data-root "$AVB_DATA_ROOT" \
  --checkpoint saved_model_topology/wingsnet_best.pth \
  --make-video \
  --make-flythrough
```

The Slurm file defaults to the project cluster's `gpu-stud` partition. On another cluster, override its resource settings when submitting or edit only the `#SBATCH` header. For example:

```bash
sbatch --partition=<gpu-partition> run_lidc_case_gpu.sh <the same arguments>
```

Without Slurm, run the same workflow directly from an activated environment:

```bash
python run_lidc_case.py \
  --case-id LIDC-IDRI-0011 \
  --series-uid "$SERIES_UID" \
  --dicom-root "$AVB_DATA_ROOT/lidc/idc_downloads" \
  --xml-root "$AVB_DATA_ROOT/lidc/lidc_idri/LIDC-XML-only" \
  --data-root "$AVB_DATA_ROOT" \
  --checkpoint saved_model_topology/wingsnet_best.pth \
  --device cuda \
  --amp \
  --make-video \
  --make-flythrough \
  --no-xvfb
```

Omit `--no-xvfb` on a headless Linux machine that has Xvfb available. Use `--device cpu` only for functional testing; full-volume inference can be very slow on CPU.

### 8. Monitor the Job

The job ID is printed by `sbatch`.

```bash
squeue -j <JOBID>
tail -f lidc_case_<JOBID>.out
cat lidc_case_<JOBID>.err
```

Completion is indicated by:

```text
LIDC case workflow complete
```

A queued Slurm job is waiting for cluster resources; it is not stuck. A job that disappears from `squeue` can be checked with:

```bash
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode,MaxRSS
```

### 9. Check the Outputs

For the example above, the canonical outputs are:

```text
$AVB_DATA_ROOT/predictions/LIDC-IDRI-0011_topology_auto/
$AVB_DATA_ROOT/centerlines/LIDC-IDRI-0011_topology_auto_thr02/
$AVB_DATA_ROOT/visualizations/LIDC-IDRI-0011_topology_auto_thr02/
```

Important files:

```text
predictions/.../pred_lumen.npy
centerlines/.../pred_lumen_centerline_pruned_graph.json
centerlines/.../pred_lumen_paths.json
centerlines/.../pred_lumen_smoothed_paths.json
centerlines/.../all_poi_target_audit.csv
centerlines/.../selected_target_audit.json
visualizations/.../airway_mesh_path_overlay.png
visualizations/.../airway_mesh_path_overlay.gif
visualizations/.../airway_flythrough_bronchoscopy.gif
visualizations/.../lidc_case_run_summary.json
```

The LIDC target is a nodule centroid, not an airway-centerline coordinate. The final navigation endpoint is therefore the closest recovered graph node and should remain inside the predicted lumen. `target_in_lumen: false` at the nodule center alone does not mean the planner failed; inspect the selected graph distance, local airway support, path occupancy, and overlay together.

### 10. Three Supported Workflows

Use the appropriate entry point rather than manually combining every script:

| Goal | Entry point |
|---|---|
| Test one new LIDC case | `sbatch run_lidc_case_gpu.sh ...` |
| Validate the final checkpoint on held-out AirRC | `sbatch final_validation_gpu.sh` |
| Compare baseline/topology/postprocessing settings | `sbatch ablation_validation_gpu.sh` |
| Rebuild AirRC patches | `sbatch extract_patches_gpu.sh` |
| Fine-tune WingsNet | `sbatch train_gpu.sh` |

The AirRC/ATM-style training data are governed by their original dataset terms and are not assumed to be bundled with this repository. Preserve the case-level train/validation split generated by `extract_airrc_patches.py`.

### 11. Troubleshooting

| Problem | Meaning and action |
|---|---|
| `torch.cuda.is_available() is False` | Run inside an allocated GPU node; then verify the installed PyTorch wheel matches the cluster CUDA runtime. |
| `nvidia-smi: command not found` on login node | The login node may not expose a GPU. Request a GPU allocation or submit with `sbatch`. |
| Slurm job remains `PENDING` | Inspect `squeue -j <JOBID> -o "%.18i %.2t %.10M %.30R"`; the final column explains the scheduling reason. |
| Multiple DICOM series found | Pass the exact CT `--series-uid`. The one-command wrapper locates it recursively. |
| XML series mismatch | Use the XML belonging to the same `SeriesInstanceUID`; do not bypass the check for normal experiments. |
| Checkpoint missing/incompatible | Place the documented checkpoint correctly. Loading is strict by default to prevent partial-model inference. |
| PyVista display/X server error | On a cluster install/use Xvfb. On a desktop session pass `--no-xvfb`. |
| `blender: command not found` | Blender is optional and separate from the Python environment; install it only for printable-shell export. |
| Output looks branched but misses the nodule region | Smoothing cannot create an absent airway. Inspect `all_poi_target_audit.csv` and the predicted lumen support near the POI. |
| Rerunning repeats expensive inference | Add `--reuse-prediction` only when the processed CT, run ID, model architecture, and checkpoint are unchanged. |

### 12. Reproducibility Notes

- Coordinates are stored as voxel `z,y,x` unless a field explicitly says physical `x,y,z`.
- Processed LIDC volumes use 1 mm isotropic spacing by default.
- DICOM/XML series identity is checked before POI conversion.
- Checkpoints load strictly unless `--allow-partial-checkpoint` is explicitly requested for controlled architecture migration.
- LIDC XML supplies nodule annotations, not airway ground truth. LIDC results are qualitative target-navigation demonstrations.
- Quantitative segmentation and branch-recall claims must come from held-out AirRC cases with airway labels.
- The flythrough is a visualization of a planned centerline path, not proof of physical bronchoscope clearance or autonomous clinical safety.


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
run_final_validation_suite.py
```

Runs the final pipeline on held-out AirRC validation cases and aggregates centerline/branch/path metrics into JSON and CSV. This is the main evidence that the checkpoint improves connected airway recovery beyond the single LIDC example.

```text
run_validation_ablation.py
```

Runs report-ready ablations comparing the baseline checkpoint, topology checkpoint, threshold choices, and pruning choices.

```text
summarize_validation_tables.py
```

Creates a Markdown/JSON results table from one or more validation summary CSV files.

```text
make_final_airway_report.py
```

Packages the final LIDC target audit, graph/path metrics, flythrough metrics, and mesh topology stats into one report. This is a reporting wrapper; it does not replace held-out validation.

```text
blender_make_hollow_airway_shell.py
```

Creates a printable hollow airway shell from the open lumen surface using Blender Solidify. Keep this separate from the navigation mesh: the open lumen mesh is for virtual bronchoscopy, while the hollow shell is for physical print testing.

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

If the folder contains more than one DICOM series, pass the exact CT `SeriesInstanceUID`:

```bash
--series-uid 1.2.840....
```

The chosen UID is written into the metadata as `selected_series_uid`. The LIDC POI extraction script checks this against the XML `SeriesInstanceUID` before converting nodule coordinates.

Example DICOM folder:

```bash
CASE_ID=LIDC-IDRI-0001

CT_DIR=/home/opat90op/AMS_Project/datasets_new/lidc/lidc_idri/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/CT_1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192
```

Run:

```bash
python preprocess_lidc_for_inference.py \
  --dicom-dir ${CT_DIR} \
  --series-uid <CT_SERIES_INSTANCE_UID> \
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

### Held-out AirRC Validation

The LIDC nodule case checks target navigation, but it does not have airway ground truth. To judge whether the model really learned deeper connected airway branches, run the final checkpoint on held-out AirRC cases:

```bash
python run_final_validation_suite.py \
  --data-root /home/opat90op/AMS_Project/datasets_new \
  --checkpoint saved_model_topology/wingsnet_best.pth \
  --device cuda \
  --limit 51 \
  --mask-threshold 0.2 \
  --skeleton-threshold 0.2 \
  --skeleton-low-threshold 0.08 \
  --prune-length 12 \
  --preserve-generations 8 \
  --output-root /home/opat90op/AMS_Project/datasets_new/final_validation/airrc
```

Outputs:

```text
/home/opat90op/AMS_Project/datasets_new/final_validation/airrc/final_validation_summary.json
/home/opat90op/AMS_Project/datasets_new/final_validation/airrc/final_validation_summary.csv
```

Important fields:

```text
gt_branch_recall_3vox_80pct
gt_branch_recall_gen6
gt_branch_recall_gen7
gt_branch_recall_gen8
gt_centerline_coverage_1vox
gt_centerline_coverage_2vox
gt_centerline_coverage_3vox
gt_centerline_coverage_5vox
pred_centerline_precision_2vox
gt_symmetric_mean_distance
gt_length_ratio_pred_over_gt
lumen_dice
lumen_precision
lumen_recall
timi_max_generation
timi_unreached_branches
endpoints
branchpoints
short_terminal_branches
first_path_inside_lumen_ratio
mean_path_inside_lumen_ratio
min_path_inside_lumen_ratio
paths_below_095_inside_lumen
```

Accept the model only if ground-truth branch recall/centerline coverage and airway depth improve without a large increase in short/noisy terminal branches. `gt_branch_recall_*` is the key branch-completeness metric; `gt_centerline_coverage_*` measures how much of the true airway centerline is recovered by the predicted skeleton.

### Ablation Comparison

For the report, compare the final method against baseline/postprocessing variants:

```bash
sbatch ablation_validation_gpu.sh
```

This runs:

```text
baseline checkpoint, threshold 0.2, prune 12
topology checkpoint, threshold 0.2, prune 12
topology checkpoint, threshold 0.2, no pruning
topology checkpoint, threshold 0.3, prune 12
```

The summary table is written to:

```text
/home/opat90op/AMS_Project/datasets_new/final_validation/ablations/ablation_summary_table.md
```

For the final report, use this table to show whether the topology-aware checkpoint and chosen cleanup settings improve branch recall without adding noisy branches.

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
