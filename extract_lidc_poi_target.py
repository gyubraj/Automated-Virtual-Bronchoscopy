from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


def read_dicom_series(dicom_dir):
    if sitk is None:
        raise RuntimeError("SimpleITK is required. Install it with: pip install SimpleITK")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))

    if series_ids:
        files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    else:
        files = reader.GetGDCMSeriesFileNames(str(dicom_dir))

    if not files:
        raise RuntimeError(f"No DICOM files found in {dicom_dir}")

    reader.SetFileNames(files)
    return reader.Execute()


def make_resampled_reference(meta):
    resampled = meta["resampled"]
    image = sitk.Image(
        [int(v) for v in resampled["size_xyz"]],
        sitk.sitkFloat32,
    )
    image.SetSpacing(tuple(float(v) for v in resampled["spacing_xyz"]))
    image.SetOrigin(tuple(float(v) for v in resampled["origin_xyz"]))
    image.SetDirection(tuple(float(v) for v in resampled["direction"]))
    return image


def strip_namespace(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


def child_text(element, name):
    for child in element:
        if strip_namespace(child.tag) == name:
            return child.text
    return None


def find_children(element, name):
    return [child for child in element if strip_namespace(child.tag) == name]


def iter_descendants(element, name):
    for child in element.iter():
        if strip_namespace(child.tag) == name:
            yield child


def parse_float(value):
    return float(value) if value not in (None, "") else None


def parse_int(value):
    return int(float(value)) if value not in (None, "") else None


def extract_nodule_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    nodules = []

    for session_index, session in enumerate(iter_descendants(root, "readingSession")):
        for nodule_index, nodule in enumerate(find_children(session, "unblindedReadNodule")):
            nodule_id = child_text(nodule, "noduleID") or f"session{session_index}_nodule{nodule_index}"
            points = []

            for roi in find_children(nodule, "roi"):
                z_position = parse_float(child_text(roi, "imageZposition"))
                if z_position is None:
                    continue

                for edge_map in find_children(roi, "edgeMap"):
                    x = parse_int(child_text(edge_map, "xCoord"))
                    y = parse_int(child_text(edge_map, "yCoord"))
                    if x is None or y is None:
                        continue
                    points.append([x, y, z_position])

            if points:
                nodules.append({
                    "reader_index": session_index,
                    "nodule_index": nodule_index,
                    "nodule_id": nodule_id,
                    "points_x_y_zpos": points,
                })

    return nodules


def physical_z_values(image):
    values = []
    for k in range(image.GetSize()[2]):
        point = image.TransformIndexToPhysicalPoint((0, 0, k))
        values.append(point[2])
    return np.asarray(values, dtype=np.float64)


def nodule_to_target(nodule, original_img, resampled_img):
    z_values = physical_z_values(original_img)
    physical_points = []

    for x, y, z_position in nodule["points_x_y_zpos"]:
        k = int(np.argmin(np.abs(z_values - z_position)))
        physical_point = original_img.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(k)))
        physical_points.append(physical_point)

    physical_points = np.asarray(physical_points, dtype=np.float64)
    centroid_physical_xyz = physical_points.mean(axis=0)
    target_cont_xyz = resampled_img.TransformPhysicalPointToContinuousIndex(tuple(centroid_physical_xyz))
    target_xyz = [int(round(v)) for v in target_cont_xyz]
    target_zyx = [target_xyz[2], target_xyz[1], target_xyz[0]]

    return {
        "nodule_id": nodule["nodule_id"],
        "reader_index": nodule["reader_index"],
        "nodule_index": nodule["nodule_index"],
        "point_count": len(nodule["points_x_y_zpos"]),
        "centroid_physical_xyz": [float(v) for v in centroid_physical_xyz],
        "target_xyz": [int(v) for v in target_xyz],
        "target_zyx": [int(v) for v in target_zyx],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract LIDC XML nodule POIs and convert them to resampled CT target z,y,x coordinates."
    )
    parser.add_argument("--xml", required=True, help="LIDC XML annotation file")
    parser.add_argument("--dicom-dir", required=True, help="Original CT DICOM folder used for preprocessing")
    parser.add_argument("--metadata", required=True, help="processed_lidc metadata JSON for the resampled CT")
    parser.add_argument("--output-json", required=True, help="Output target POI JSON")
    args = parser.parse_args()

    if sitk is None:
        raise RuntimeError("SimpleITK is required. Install it with: pip install SimpleITK")

    with open(args.metadata, "r") as f:
        meta = json.load(f)

    original_img = read_dicom_series(args.dicom_dir)
    resampled_img = make_resampled_reference(meta)
    nodules = extract_nodule_annotations(args.xml)

    targets = [
        nodule_to_target(nodule, original_img, resampled_img)
        for nodule in nodules
    ]

    payload = {
        "xml": str(Path(args.xml)),
        "dicom_dir": str(Path(args.dicom_dir)),
        "metadata": str(Path(args.metadata)),
        "target_count": len(targets),
        "targets": targets,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("Done")
    print("  targets:", len(targets))
    print("  output:", output_json)
    for index, target in enumerate(targets[:10]):
        print(
            f"  [{index}] nodule_id={target['nodule_id']} "
            f"points={target['point_count']} target_zyx={target['target_zyx']}"
        )


if __name__ == "__main__":
    main()
