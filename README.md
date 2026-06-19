# Automated Virtual Bronchoscopy

Airway segmentation and virtual bronchoscopy pipeline based on WingsNet. The repo can preprocess AirRC/LIDC data, train a lumen/wall model, run full-volume inference, extract a centerline graph, plan airway paths, and export mesh/overlay visualizations.

## Setup

```bash
git clone <repo-url>
cd Automated-Virtual-Bronchoscopy
python -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy scikit-image SimpleITK pyvista nibabel
```

Install the CUDA build of PyTorch that matches your GPU/driver if you train or infer on GPU.

Expected files:

- Model checkpoint: `saved_model/wingsnet_best.pth`
- Training/preprocessing data root used by scripts: `~/AMS_Project/datasets_new`
- AirRC labels: `~/AMS_Project/datasets_new/airrc/labelsTr/*.nii.gz`
- LIDC DICOM folders: `~/AMS_Project/datasets_new/lidc/lidc_idri/**/CT_<case_id>`

## Run Full Pipeline

Use this for CT to prediction, centerline, path, mesh, and validation image. Inputs can be processed `.npy`, `.nii/.nii.gz`, or a DICOM folder. Add `--dry-run` to print the commands without running them. Outputs are written under:

- `predictions/<case_id>/`
- `centerlines/<case_id>_thr*/`
- `visualizations/<case_id>_thr*/`

CPU example used for one AirRC case:

```bash
CASE=1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860
DATA_ROOT=/home/opat90op/AMS_Project/datasets_new

python run_airway_pipeline.py \
  --ct "$DATA_ROOT/processed_airrc/images/${CASE}_ct.npy" \
  --case-id "$CASE" \
  --data-root "$DATA_ROOT" \
  --checkpoint saved_model/wingsnet_best.pth \
  --model-module WingsNet \
  --model-class WingsNet \
  --model-kwargs '{"in_channel": 1, "n_classes": 2}' \
  --device cpu \
  --mask-threshold 0.5 \
  --skeleton-threshold 0.2 \
  --prune-length 12 \
  --root-mode timi-trachea \
  --trachea-root-end min-z \
  --target-mode farthest-endpoint \
  --spacing 1,1,1 \
  --make-video
```

LIDC example after preprocessing the DICOM series to `.npy`:

```bash
CASE=LIDC-IDRI-0001_resampled_fixed
DATA_ROOT=/home/opat90op/AMS_Project/datasets_new

python run_airway_pipeline.py \
  --ct "$DATA_ROOT/processed_lidc/images/LIDC-IDRI-0001_ct.npy" \
  --case-id "$CASE" \
  --data-root "$DATA_ROOT" \
  --checkpoint saved_model/wingsnet_best.pth \
  --model-module WingsNet \
  --model-class WingsNet \
  --model-kwargs '{"in_channel": 1, "n_classes": 2}' \
  --device cpu \
  --no-normalize \
  --mask-threshold 0.5 \
  --skeleton-threshold 0.2 \
  --prune-length 12 \
  --root-mode timi-trachea \
  --trachea-root-end min-z \
  --target-mode farthest-endpoint \
  --spacing 1,1,1 \
  --make-video
```

Use normal double hyphens in command flags. For example, type `--save-binary` and `--skeleton-metrics`, not `—save-binary` or `—skeleton-metrics`.

## Run Steps Manually

Preprocess AirRC labels and matching LIDC CT DICOMs:

```bash
python preprocess_airrc.py
python verify_processed_airrc.py
```

Extract training patches:

```bash
python extract_airrc_patches.py
```

Train WingsNet:

```bash
python train.py
```

On SLURM:

```bash
sbatch train_gpu.sh
```

Edit the `cd` path in `train_gpu.sh` before submitting on a new machine.

Run full-volume inference only:

```bash
python infer_wingsnet_full_volume.py \
  --checkpoint saved_model/wingsnet_best.pth \
  --ct "$DATA_ROOT/processed_airrc/images/${CASE}_ct.npy" \
  --output-dir "$DATA_ROOT/predictions/$CASE" \
  --model-module WingsNet \
  --model-class WingsNet \
  --model-kwargs '{"in_channel": 1, "n_classes": 2}' \
  --device cpu \
  --threshold 0.5 \
  --save-binary
```

For already-normalized LIDC `.npy` output from `preprocess_lidc_for_inference.py`, add `--no-normalize`:

```bash
python infer_wingsnet_full_volume.py \
  --checkpoint saved_model/wingsnet_best.pth \
  --ct "$DATA_ROOT/processed_lidc/images/LIDC-IDRI-0001_ct.npy" \
  --output-dir "$DATA_ROOT/predictions/LIDC-IDRI-0001_resampled_fixed" \
  --model-module WingsNet \
  --model-class WingsNet \
  --model-kwargs '{"in_channel": 1, "n_classes": 2}' \
  --device cpu \
  --no-normalize \
  --threshold 0.5 \
  --save-binary
```

Skeletonize lumen and plan paths from the trachea:

```bash
python skeletonize_airrc_case.py \
  --input "$DATA_ROOT/predictions/$CASE/pred_lumen.npy" \
  --output-dir "$DATA_ROOT/centerlines/${CASE}_thr02" \
  --threshold 0.2 \
  --keep-largest \
  --prune-length 12 \
  --root-mode timi-trachea \
  --trachea-root-end min-z \
  --target-mode farthest-endpoint
```

For LIDC XML nodule POI planning, use `--target-zyx` instead of `--target-mode farthest-endpoint`:

```bash
python skeletonize_airrc_case.py \
  --input "$DATA_ROOT/predictions/LIDC-IDRI-0001_resampled_fixed/pred_lumen.npy" \
  --output-dir "$DATA_ROOT/centerlines/LIDC-IDRI-0001_resampled_fixed_thr02_poi_main" \
  --threshold 0.2 \
  --keep-largest \
  --prune-length 12 \
  --root-mode timi-trachea \
  --trachea-root-end min-z \
  --target-zyx 224,257,222
```

Export airway mesh and path overlay:

```bash
python visualizer_airway_mesh_path.py \
  --mask "$DATA_ROOT/predictions/$CASE/pred_lumen.npy" \
  --paths-json "$DATA_ROOT/centerlines/${CASE}_thr02/pred_lumen_paths.json" \
  --output-dir "$DATA_ROOT/visualizations/${CASE}_thr02" \
  --threshold 0.2 \
  --tube-radius 0.8 \
  --endpoint-point-size 6 \
  --make-video
```

Export a smoother visual mesh using TIMI-style largest-component/hole-fill cleanup plus mesh smoothing:

```bash
python visualizer_airway_mesh_path.py \
  --mask "$DATA_ROOT/predictions/LIDC-IDRI-0001_resampled_fixed/pred_lumen.npy" \
  --paths-json "$DATA_ROOT/centerlines/LIDC-IDRI-0001_resampled_fixed_thr02_poi_main/pred_lumen_paths.json" \
  --output-dir "$DATA_ROOT/visualizations/LIDC-IDRI-0001_resampled_fixed_thr02_poi_main_smooth" \
  --threshold 0.2 \
  --tube-radius 0.8 \
  --endpoint-point-size 6 \
  --mesh-keep-largest \
  --mesh-fill-holes \
  --mesh-closing-radius 1 \
  --mesh-smooth-iterations 15 \
  --make-video
```

Evaluate with an optional target:

```bash
python evaluation.py \
  --checkpoint saved_model/wingsnet_best.pth \
  --ct "$DATA_ROOT/processed_airrc/images/${CASE}_ct.npy" \
  --target "$DATA_ROOT/processed_airrc/targets/${CASE}_target.npy" \
  --output-dir "$DATA_ROOT/predictions/$CASE" \
  --model-module WingsNet \
  --model-class WingsNet \
  --model-kwargs '{"in_channel": 1, "n_classes": 2}' \
  --device cpu \
  --skeleton-metrics
```

Extract LIDC XML POI targets:

```bash
CASE_ID=LIDC-IDRI-0001
CT_DIR="$DATA_ROOT/lidc/lidc_idri/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/CT_1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192"
XML="$DATA_ROOT/lidc/lidc_idri/LIDC-XML-only/tcia-lidc-xml/185/069.xml"

python extract_lidc_poi_target.py \
  --xml "$XML" \
  --dicom-dir "$CT_DIR" \
  --metadata "$DATA_ROOT/processed_lidc/metadata/${CASE_ID}_meta.json" \
  --output-json "$DATA_ROOT/processed_lidc/metadata/${CASE_ID}_poi_targets.json"
```

Validate a centerline graph:

```bash
python validate_centerline_graph.py \
  --pred-lumen "$DATA_ROOT/predictions/$CASE/pred_lumen.npy" \
  --skeleton "$DATA_ROOT/centerlines/${CASE}_thr02/pred_lumen_centerline_pruned.npy" \
  --graph-json "$DATA_ROOT/centerlines/${CASE}_thr02/pred_lumen_centerline_pruned_graph.json" \
  --paths-json "$DATA_ROOT/centerlines/${CASE}_thr02/pred_lumen_paths.json" \
  --threshold 0.2 \
  --short-branch-length 12 \
  --timi-tree-parse \
  --output-json "$DATA_ROOT/centerlines/${CASE}_thr02/centerline_validation_metrics.json"
```

For AirRC cases with ground-truth target masks, add:

```bash
--target "$DATA_ROOT/processed_airrc/targets/${CASE}_target.npy"
```

## Main Scripts

- `run_airway_pipeline.py`: one-command inference, skeleton, path, mesh, and overlay pipeline.
- `preprocess_airrc.py`: resamples CT to labels and saves normalized CT plus 2-channel targets.
- `extract_airrc_patches.py`: creates `128x128x128` training patches and split JSON files.
- `train.py`: current WingsNet training script for AirRC patches.
- `infer_wingsnet_full_volume.py`: sliding-window full-volume model inference.
- `skeletonize_airrc_case.py`: centerline skeleton, graph pruning, and path planning.
- `visualizer_airway_mesh_path.py`: exports STL/OBJ mesh and PNG/GIF validation render.
- `evaluation.py`: inference plus voxel and optional skeleton metrics.
- `validate_centerline_graph.py`: validates predicted centerline graphs and planned paths.
- `extract_lidc_poi_target.py`: converts LIDC XML nodule annotations to `target_zyx` coordinates for POI-driven path planning.
