import copy
import os
import os.path as osp
from pathlib import Path

import numpy as np
from bop_toolkit_lib import inout
from tqdm import tqdm

from src.utils.dataset import LMO_index_to_ID
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_root_project():
    return Path(__file__).absolute().parent.parent.parent


def combine(list_dict):
    output = {}
    for dict_ in list_dict:
        for field in dict_.keys():
            for name_data in dict_[field].keys():
                key = field + "_" + name_data
                assert key not in output.keys()
                output[key] = dict_[field][name_data]
    return output


def group_by_image_level(data, image_key="im_id"):
    data_per_image = {}
    for det in data:
        dets = [det] if isinstance(det, dict) else det
        for det in dets:
            scene_id, im_id = int(det["scene_id"]), int(det[image_key])
            key = f"{scene_id:06d}_{im_id:06d}"
            data_per_image.setdefault(key, []).append(det)
    return data_per_image


def save_bop_results(path, results, additional_name=None):
    if additional_name is not None:
        lines = [f"scene_id,im_id,obj_id,score,R,t,time,{additional_name}"]
    else:
        lines = ["scene_id,im_id,obj_id,score,R,t,time"]

    for res in results:
        line = "{scene_id},{im_id},{obj_id},{score},{R},{t},{time}".format(
            scene_id=res["scene_id"],
            im_id=res["im_id"],
            obj_id=res["obj_id"],
            score=res["score"],
            R=" ".join(map(str, res["R"].flatten().tolist())),
            t=" ".join(map(str, res["t"].flatten().tolist())),
            time=res.get("time", -1),
        )
        if additional_name is not None:
            line += f",{res[additional_name]}"
        lines.append(line)

    with open(path, "w") as f:
        f.write("\n".join(lines))


def calculate_runtime_per_image(results):
    times = {}
    for result in results:
        result_key = f"{result['scene_id']:06d}_{result['im_id']:06d}"
        if result_key not in times:
            times[result_key] = {"time": [], "batch_id": [], "detection_time": 0}

        if result["batch_id"] not in times[result_key]["batch_id"]:
            times[result_key]["batch_id"].append(result["batch_id"])
            times[result_key]["time"].append(result["time"])
            times[result_key]["detection_time"] = result["additional_time"]

        del result["additional_time"]
        del result["batch_id"]

    total_run_times = {
        key: time["detection_time"] + np.sum(time["time"])
        for key, time in times.items()
    }

    average_run_times = []
    for result in results:
        result_key = f"{result['scene_id']:06d}_{result['im_id']:06d}"
        result["time"] = total_run_times[result_key]
        average_run_times.append(result["time"])
    logger.info(f"Average runtime per image: {np.mean(average_run_times):.3f} s")
    return results


def save_predictions_from_batched_predictions(
    prediction_dir,
    dataset_name,
    model_name,
    run_id,
):
    list_files = sorted(
        file for file in os.listdir(prediction_dir) if file.endswith(".npz")
    )

    top1_predictions, topk_predictions = [], []
    instance_id = 0

    for batch_id, file in tqdm(
        enumerate(list_files), desc="Formatting predictions ..."
    ):
        data = np.load(osp.join(prediction_dir, file))
        assert len(data["poses"].shape) in [3, 4]
        is_only_top1 = len(data["poses"].shape) == 3
        if not is_only_top1:
            k = data["poses"].shape[1]

        for idx_sample in range(len(data["im_id"])):
            obj_id = int(data["object_id"][idx_sample])
            if "lmo" in dataset_name:
                obj_id = LMO_index_to_ID[obj_id - 1]

            if is_only_top1:
                t = data["poses"][idx_sample][:3, 3].reshape(-1)
                R = data["poses"][idx_sample][:3, :3].reshape(-1)
                score = data["scores"][idx_sample]
            else:
                t = data["poses"][idx_sample][0][:3, 3].reshape(-1)
                R = data["poses"][idx_sample][0][:3, :3].reshape(-1)
                score = data["scores"][idx_sample][0]

            top1_prediction = dict(
                scene_id=int(data["scene_id"][idx_sample]),
                im_id=int(data["im_id"][idx_sample]),
                obj_id=obj_id,
                score=score,
                t=t,
                R=R,
                time=data["time"][idx_sample],
                additional_time=data["detection_time"][idx_sample],
                batch_id=np.copy(batch_id),
            )
            top1_predictions.append(top1_prediction.copy())
            topk_predictions.append({**top1_prediction, "instance_id": instance_id})

            if not is_only_top1:
                for idx_k in range(1, k):
                    topk_predictions.append(
                        dict(
                            scene_id=int(data["scene_id"][idx_sample]),
                            im_id=int(data["im_id"][idx_sample]),
                            obj_id=obj_id,
                            score=data["scores"][idx_sample][idx_k],
                            t=data["poses"][idx_sample][idx_k][:3, 3].reshape(-1),
                            R=data["poses"][idx_sample][idx_k][:3, :3].reshape(-1),
                            time=data["time"][idx_sample],
                            instance_id=instance_id,
                            additional_time=data["detection_time"][idx_sample],
                            batch_id=np.copy(batch_id),
                        )
                    )
            instance_id += 1

    name_file = f"{model_name}-pbrreal-rgb-mmodel_{dataset_name}-test_{run_id}"
    save_path = osp.join(prediction_dir, f"{name_file}.csv")
    calculate_runtime_per_image(top1_predictions)
    save_bop_results(save_path, top1_predictions)
    logger.info(f"Saved predictions to {save_path}")

    if not is_only_top1:
        save_path = osp.join(prediction_dir, f"{name_file}MultiHypothesis.csv")
        calculate_runtime_per_image(topk_predictions)
        save_bop_results(
            save_path,
            topk_predictions,
            additional_name="instance_id",
        )
        logger.info(f"Saved predictions to {save_path}")


def generate_test_list(all_detections):
    all_target_list = {}
    for im_key, im_dets in all_detections.items():
        im_id, scene_id = im_key.split("_")
        im_id, scene_id = int(im_id), int(scene_id)
        im_target = {}
        for det in im_dets:
            if "category_id" in det:
                obj_id = det["category_id"]
            elif "obj_id" in det:
                obj_id = det["obj_id"]
            else:
                raise ValueError("category_id or obj_id is not in the detection")
            im_target[obj_id] = im_target.get(obj_id, 0) + 1

        all_target_list[im_key] = [
            {
                "scene_id": scene_id,
                "im_id": im_id,
                "obj_id": obj_id,
                "inst_count": inst_count,
            }
            for obj_id, inst_count in im_target.items()
        ]
    return all_target_list


def load_test_list_and_cnos_detections(
    root_dir, dataset_name, test_setting, max_det_per_object_id=None
):
    if dataset_name in ["lmo", "tless", "tudl", "icbin", "itodd", "hb", "ycbv", "T282"]:
        year = "19"
        det_model = "cnos-fastsam"
    elif dataset_name in ["hope"]:
        year = "24"
        det_model = "cnos-sam"
    else:
        raise NotImplementedError(
            f"Dataset {dataset_name} is not supported with default detections"
        )

    cnos_dets_dir = (
        root_dir / "default_detections" / f"core{year}_model_based_unseen/" / det_model
    )
    avail_det_files = os.listdir(cnos_dets_dir)
    cnos_dets_path = [file for file in avail_det_files if dataset_name in file][0]
    all_cnos_dets = inout.load_json(os.path.join(cnos_dets_dir, cnos_dets_path))
    all_cnos_dets_per_image = group_by_image_level(all_cnos_dets, image_key="image_id")

    if test_setting == "detection":
        return generate_test_list(all_cnos_dets_per_image), all_cnos_dets_per_image

    if test_setting != "localization":
        raise NotImplementedError(f"Test setting {test_setting} is not supported")

    target_file_path = root_dir / dataset_name / f"test_targets_bop{year}.json"
    assert target_file_path.exists(), (
        "Combination (dataset, test_setting, year)="
        f"{dataset_name, test_setting, year} is not available"
    )
    logger.info(f"Loading test list from {target_file_path}")
    test_list = inout.load_json(target_file_path)
    selected_detections = []
    for test in tqdm(test_list):
        test_object_id = test["obj_id"]
        scene_id, im_id = test["scene_id"], test["im_id"]
        image_key = f"{scene_id:06d}_{im_id:06d}"

        if image_key not in all_cnos_dets_per_image:
            logger.info(f"No detection for {image_key}")
            continue

        cnos_dets_per_image = all_cnos_dets_per_image[image_key]
        dets = [
            det for det in cnos_dets_per_image if det["category_id"] == test_object_id
        ]
        if len(dets) == 0:
            dets = copy.deepcopy(cnos_dets_per_image)
            for det in dets:
                det["category_id"] = test_object_id

        assert len(dets) > 0
        dets = sorted(dets, key=lambda x: x["score"], reverse=True)
        num_instances = max_det_per_object_id or test["inst_count"]
        selected_detections.append(dets[:num_instances])

    logger.info(f"Detections: {len(test_list)} test samples")
    assert len(selected_detections) == len(test_list)
    return (
        group_by_image_level(test_list, image_key="im_id"),
        group_by_image_level(selected_detections, image_key="image_id"),
    )
