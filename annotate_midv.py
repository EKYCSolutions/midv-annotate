import os
import json
import argparse
from glob import glob
#
import cv2
import numpy as np
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
#
from utils import determine_reference_axis_from_polygon, calculate_angle_between_two_line_segments
#
parser = argparse.ArgumentParser(
    description="Using Facebook's SAM2 model to add mask, bbox, rotation angel")
parser.add_argument('data_dir', type=str,
                    help='Path to a directory containing all zip files downloaded from MIDV500/MIDV2019')
parser.add_argument('--labelmaps_file', type=str,
                    help='Path to labelmaps file', default='labelmaps.example.json')
parser.add_argument('--output_dir', type=str,
                    help='path to the output folder.', default="out")
parser.add_argument('--sam2_checkpoint', type=str,
                    help='path to sam2 checkpoint file', default="sam2_hiera_tiny.pt")
parser.add_argument('--sam2_config', type=str,
                    help='path to sam2 config file', default="sam2_hiera_t.yaml")
parser.add_argument('--bbox_visibility_width', type=float,
                    help="The percentage of the bbox's width needed to be visible for the sample to be annotated", default=0.2)
parser.add_argument('--smoothing_strength', type=float,
                    help="The percentage of the largest contour longest arc to be used as gaps between boundary markers.", default=0.0005)
parser.add_argument('--num_key_pts_to_sample', type=int,
                    help="The number of key points to sample within the annotated quad to provide to SAM2", default=10)
args = parser.parse_args()


def annotate_document(image, bbox, key_pts):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([key_pts]),
        point_labels=np.array([[1]*len(key_pts)]),
        box=bbox[None, :],
        multimask_output=False,
    )
    sorted_ind = np.argsort(scores)[::-1]
    mask = masks[sorted_ind][0].astype(np.uint8)*255
    #
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    # select contour with the largest number of points
    contour = sorted(contours, key=lambda x: len(x), reverse=True)[0]

    # smooth out the resulting polygons by reducing the number of points
    epsilon = args.smoothing_strength * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
    contour = contour[:, 0, :]

    return contour.tolist()


def load_processed_files():
    with open(os.path.join(args.output_dir, "checkpoints.txt"), "r") as checkpoint_files:
        return [i.rstrip("\n").split("\t") for i in checkpoint_files.readlines()]


if __name__ == "__main__":
    DATA_DIR = "/tmp/midv500"
    #

    SAMPLES_DIR = os.path.join(args.output_dir, "samples")
    CHECKPOINT_FILE_PATH = os.path.join(args.output_dir, "checkpoints.txt")
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    print("Loading SAM2")
    sam2_model = build_sam2(
        args.sam2_config,
        args.sam2_checkpoint,
        device='cpu'
    )
    predictor = SAM2ImagePredictor(sam2_model)

    print(f"Loading {args.labelmaps_file}")
    with open(os.path.join(args.labelmaps_file), "r") as labelmaps_file:
        labelmaps = json.load(labelmaps_file)

    print("Creating labelmaps file")
    with open(os.path.join(args.output_dir, "labelmaps.txt"), "w") as labelmaps_file:
        out = sorted(set([v for _, v in labelmaps.items()]))
        out = list(out)
        labelmaps_file.write("\n".join(out))

    print("Determining starting point")
    all_zip_files = sorted(glob(os.path.join(args.data_dir, "*.zip")))
    all_zip_names = [os.path.basename(i).replace(
        ".zip", "") for i in all_zip_files]

    if os.path.exists(CHECKPOINT_FILE_PATH):
        print(f"\t├──  checkpoint file exists")
        with open(os.path.join(args.output_dir, "checkpoints.txt"), "r") as checkpoints_file:
            checkpoints = [i.rstrip("\n")
                           for i in checkpoints_file.readlines()]
        print(
            f"\t├──  starting from {all_zip_names[all_zip_names.index(checkpoints[-1])+1]}")
    else:
        print(f"\t├──  starting from scratch")
        checkpoints = []

    remaining = [i for i in all_zip_files if os.path.basename(
        i).replace(".zip", "") not in checkpoints]

    for zip_filepath in remaining:
        dirname = os.path.basename(zip_filepath).replace(".zip", "")
        print(f"Annotating: {dirname}")
        print(f"\t├──  unzip {zip_filepath}")
        os.system(f"unzip -qq -o {zip_filepath} -d {DATA_DIR}")
        print(f"\t├──  clearing out unneccasry data to save space")
        os.remove(os.path.join(DATA_DIR, dirname, "images", dirname+".tif"))
        os.remove(os.path.join(DATA_DIR, dirname,
                  "ground_truth", dirname+".json"))
        os.system(f"rm -rf {os.path.join(DATA_DIR, dirname, 'videos')}")

        print("\t├──  Loading samples")
        images = sorted(
            glob(os.path.join(DATA_DIR, "*", "images", "*", "*.tif")))
        quads = sorted(
            glob(os.path.join(DATA_DIR, "*", "ground_truth", "*", "*.json")))

        all_samples = list(zip(images, quads))

        pbar = tqdm(initial=0, total=len(all_samples), position=1)
        cbar = tqdm(total=0, position=0, bar_format='{desc}')

        for image, quad in all_samples:
            try:
                doc_type = image\
                    .replace(DATA_DIR, "")\
                    .split(os.sep)[1]
                #
                image_filename = (
                    doc_type + "-" + os.path.basename(image)).replace(".tif", ".jpg")
                quad_filename = doc_type + "-" + os.path.basename(quad)

                # if the image already exists, skip
                i = os.path.join(args.output_dir, "samples", image_filename)
                l = os.path.join(args.output_dir, "samples", quad_filename)
                if os.path.exists(i) and os.path.exists(l):
                    pbar.update(1)
                    pbar.refresh()
                    continue

                #
                doc_type = labelmaps[doc_type]
                #
                cbar.set_description_str(
                    f"Annotating: {image_filename}, {doc_type}")
                cbar.refresh()

                image = cv2.imread(image)

                with open(quad, "r") as json_file:
                    polygon = json.load(json_file)["quad"]

                p = np.array(polygon)

                if (p[:, 0] < 0).all() or (p[:, 0] > image.shape[1]).all() or (p[:, 1] < 0).all() or (p[:, 1] > image.shape[0]).all():
                    pbar.write(
                        f"\t├──  {image_filename}'s polygon lies outside of the image, skipped")
                    pbar.update(1)
                    pbar.refresh()
                    continue

                bbox = np.zeros((4,))
                bbox[0] = max(p[:, 0].min(), 0)
                bbox[1] = max(p[:, 1].min(), 0)
                bbox[2] = min(p[:, 0].max(), image.shape[1])
                bbox[3] = min(p[:, 1].max(), image.shape[0])

                bbox_width = (bbox[2]-bbox[0])

                if (bbox_width/image.shape[0] < args.bbox_visibility_width):
                    pbar.write(
                        f"\t├──  {image_filename}'s polygon too small, skip")
                    pbar.update(1)
                    pbar.refresh()
                    continue

                x1, y1, x2, y2, x3, y3 = determine_reference_axis_from_polygon(
                    polygon,
                    line_length=image.shape[1]*0.10
                )

                parallel_marker_horizontal = np.array([
                    [x1, y1],
                    [x2, y2],
                ])

                parallel_marker_vertical = np.array([
                    [x1, y1],
                    [x3, y3],
                ])

                angle_from_horizontal = calculate_angle_between_two_line_segments(
                    parallel_marker_horizontal,
                    # horizontal line
                    np.array([
                        parallel_marker_horizontal[0],
                        [parallel_marker_horizontal[0][0]+10,
                            parallel_marker_horizontal[0][1]]
                    ])
                )

                angle_from_vertical = calculate_angle_between_two_line_segments(
                    parallel_marker_vertical,
                    # vertical line
                    np.array([
                        parallel_marker_vertical[0],
                        [parallel_marker_vertical[0][0],
                            parallel_marker_vertical[0][1]-10]
                    ])
                )

                key_pts_x = np.random.uniform(
                    p[:, 0].min(),
                    p[:, 0].max(),
                    (args.num_key_pts_to_sample, 1)
                )
                key_pts_y = np.random.uniform(
                    p[:, 1].min(),
                    p[:, 1].max(),
                    (args.num_key_pts_to_sample, 1)
                )
                key_pts = np.concatenate([
                    key_pts_x,
                    key_pts_y,
                ], axis=-1)

                boundary = annotate_document(image, bbox, key_pts)

                label = {
                    "imagePath": image_filename,
                    "imageData": None,
                    "imageHeight": image.shape[0],
                    "imageWidth": image.shape[1],
                    "shapes": [
                        {
                            "label": f"{doc_type}_polygon",
                            "points": polygon,
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                        },
                        {
                            "label": f"{doc_type}",
                            "points": boundary,
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                        },
                        {
                            "label": f"{doc_type}_bbox",
                            "points": [
                                [bbox[0], bbox[1]],
                                [bbox[2], bbox[1]],
                                [bbox[2], bbox[3]],
                                [bbox[0], bbox[3]],
                            ],
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                        },
                        {
                            "label": f"{doc_type}_rotation_marker_horizontal",
                            "rotation": angle_from_horizontal,
                            "points": parallel_marker_horizontal.tolist(),
                        },
                        {
                            "label": f"{doc_type}_rotation_marker_vertical",
                            "rotation": angle_from_vertical,
                            "points": parallel_marker_vertical.tolist(),
                        },
                    ],
                    "classes": [
                        doc_type
                    ],
                }
                #
                cv2.imwrite(os.path.join(SAMPLES_DIR, image_filename), image)
                #
                with open(os.path.join(SAMPLES_DIR, quad_filename), "w") as label_file:
                    json.dump(label, label_file)
                #
                pbar.update(1)
                pbar.refresh()
            except KeyboardInterrupt as e:
                pbar.write("KeyboardInterrupt")
                pbar.write(f"\t├──  performing cleanup")
                os.system(f"rm -rf {DATA_DIR}")
                exit()
            except Exception as e:
                print(e)
                continue

        # saving checkpoints to skip the processed zip file
        with open(os.path.join(args.output_dir, "checkpoints.txt"), "w") as checkpoints_file:
            checkpoints += [dirname]
            checkpoints_file.write("\n".join(checkpoints))

        pbar.write(f"\t├──  annotation Completed for {dirname}")
        pbar.write(f"\t├──  removing the directory")
        os.system(f"rm -rf {os.path.join(DATA_DIR, dirname)}")

        pbar.close()
        cbar.close()
