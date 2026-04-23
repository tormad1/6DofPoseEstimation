import hashlib
import json
import os
import os.path as osp
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from src.utils.logging import get_logger
from src.utils.batch import BatchedData
from src.utils.time import Timer
from src.models.poses import ObjectPoseRecovery
import src.utils.tensor_collection as tc
from src.utils.inout import save_predictions_from_batched_predictions

logger = get_logger(__name__)


class GigaPose(torch.nn.Module):
    def __init__(
        self,
        model_name,
        ae_net,
        ist_net,
        testing_metric,
        log_dir,
        max_num_dets_per_forward=None,
        template_cache_dir=None,
        template_cache_enabled=True,
        **kwargs,
    ):
        # define the network
        super().__init__()
        self.model_name = model_name
        self.ae_net = ae_net
        self.ist_net = ist_net
        self.testing_metric = testing_metric

        self.max_num_dets_per_forward = max_num_dets_per_forward
        self.template_cache_dir = (
            Path(template_cache_dir) if template_cache_dir is not None else None
        )
        self.template_cache_enabled = bool(template_cache_enabled)
        self.checkpoint_cache_key = None

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(osp.join(self.log_dir, "predictions"), exist_ok=True)

        # for testing
        self.template_datas = {}
        self.pose_recovery = {}
        self.timer = Timer()
        self.run_id = None
        self.template_datasets = None
        self.test_dataset_name = None

        logger.info("Initialize GigaPose done!")

    @property
    def device(self):
        return next(self.parameters()).device

    def set_template_data(self, dataset_name):
        template_dataset = self.template_datasets[dataset_name]
        cache_meta = self._template_cache_metadata(dataset_name, template_dataset)
        cache_path = self._template_cache_path(dataset_name, cache_meta)
        if self._load_template_cache(dataset_name, cache_path, cache_meta):
            return

        logger.info("Initializing template data ...")
        self.timer.tic()
        names = ["mask", "K", "M", "poses", "ae_features", "ist_features"]
        template_data = {name: BatchedData(None) for name in names}

        with torch.inference_mode():
            for idx in tqdm(range(len(template_dataset))):
                sample = template_dataset[idx]
                templates = sample.rgb.to(self.device)
                template_data["ae_features"].append(self.ae_net(templates))
                template_data["ist_features"].append(
                    self.ist_net.forward_by_chunk(templates)
                )

                for name in ("mask", "K", "M", "poses"):
                    template_data[name].append(getattr(sample, name).to(self.device))

        for name in names:
            template_data[name].stack()
            template_data[name] = template_data[name].data

        self._register_template_data(dataset_name, template_data)
        self._save_template_cache(cache_path, cache_meta, template_data)
        num_obj = len(template_data["K"])
        onboarding_time = self.timer.toc() / num_obj
        self.timer.reset()
        logger.info(f"Init {dataset_name} done! Avg time={onboarding_time} s/object")

    def warmup_templates(self, dataset_name=None):
        dataset_name = dataset_name or self.test_dataset_name
        if dataset_name is None:
            raise ValueError("dataset_name is required before warming up templates")
        self.set_template_data(dataset_name)

    def _register_template_data(self, dataset_name, template_data):
        self.template_datas[dataset_name] = tc.PandasTensorCollection(
            infos=pd.DataFrame(), **template_data
        )
        self.pose_recovery[dataset_name] = ObjectPoseRecovery(
            template_K=template_data["K"],
            template_Ms=template_data["M"],
            template_poses=template_data["poses"],
        )

    def _template_cache_metadata(self, dataset_name, template_dataset):
        object_templates = template_dataset.template_dataset.list_object_templates
        template_items = [
            {
                "label": template.label,
                "num_templates": template.num_templates,
                "pose_path": template.pose_path,
                "template_dir": template.template_dir,
            }
            for template in object_templates
        ]
        crop_transform = getattr(template_dataset.transforms, "crop_transform", None)
        return {
            "version": 1,
            "dataset_name": dataset_name,
            "model_name": self.model_name,
            "ae_model_name": getattr(self.ae_net, "model_name", None),
            "ist_model_name": getattr(self.ist_net, "model_name", None),
            "checkpoint": self.checkpoint_cache_key,
            "crop_target_size": getattr(crop_transform, "target_size", None),
            "patch_size": getattr(self.ae_net, "patch_size", None),
            "objects": template_items,
        }

    def _template_cache_path(self, dataset_name, cache_meta):
        if not self.template_cache_enabled or self.template_cache_dir is None:
            return None
        encoded_meta = json.dumps(cache_meta, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(encoded_meta).hexdigest()[:16]
        return self.template_cache_dir / f"{dataset_name}_{self.model_name}_{digest}.pt"

    def _load_template_cache(self, dataset_name, cache_path, cache_meta):
        if cache_path is None or not cache_path.exists():
            return False
        try:
            try:
                payload = torch.load(
                    cache_path,
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:
                payload = torch.load(cache_path, map_location=self.device)
            if payload.get("meta") != cache_meta:
                return False
            template_data = {
                name: tensor.to(self.device)
                for name, tensor in payload["tensors"].items()
            }
            self._register_template_data(dataset_name, template_data)
            logger.info(f"Loaded template data cache from {cache_path}")
            return True
        except Exception as exc:
            logger.warning(f"Failed to load template cache {cache_path}: {exc}")
            return False

    def _save_template_cache(self, cache_path, cache_meta, template_data):
        if cache_path is None:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tensors = {
                name: tensor.detach().cpu()
                for name, tensor in template_data.items()
            }
            torch.save({"meta": cache_meta, "tensors": tensors}, cache_path)
            logger.info(f"Saved template data cache to {cache_path}")
        except Exception as exc:
            logger.warning(f"Failed to save template cache {cache_path}: {exc}")

    def attach_test_metadata(
        self,
        predictions,
        test_list,
        time,
        keep_only_testing_instances=True,
    ):
        labels = np.asarray(predictions.infos.label).astype(np.int32)
        assert len(np.unique(labels)) == len(np.unique(test_list.infos.obj_id))
        detection_times = []
        if keep_only_testing_instances:
            selected_idxs = []
            for idx, id in enumerate(test_list.infos.obj_id):
                num_inst = test_list.infos.inst_count[idx]
                idx_inst = labels == id
                pred_inst = predictions[idx_inst.tolist()]

                selected_idx = torch.argsort(pred_inst.scores[:, 0], descending=True)
                selected_idx = selected_idx[:num_inst].cpu().numpy()
                selected_idx = np.arange(len(labels))[idx_inst][selected_idx]
                selected_idxs.extend(selected_idx.tolist())
                detection_times.extend(
                    [test_list.infos.detection_time[idx] for _ in range(num_inst)]
                )
        else:
            selected_idxs = np.arange(len(labels)).tolist()
            detection_times = [0 for _ in selected_idxs]
        predictions = predictions[selected_idxs]

        detection_times = np.array(detection_times)
        detection_times = torch.from_numpy(detection_times).to(
            predictions.scores.device
        )
        time = torch.ones_like(detection_times) * time
        predictions.register_tensor("detection_time", detection_times)
        predictions.register_tensor("time", time)
        return selected_idxs, predictions

    def save_batch_predictions(self, predictions, save_path):
        scene_id = np.asarray(predictions.infos.scene_id).astype(np.int32)
        im_id = np.asarray(predictions.infos.view_id).astype(np.int32)
        label = np.asarray(predictions.infos.label).astype(np.int32)
        poses = predictions.pred_poses
        scores = predictions.scores

        if poses.ndim == 4 and poses.shape[1] == 1:
            poses = poses[:, 0]
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]

        np.savez(
            save_path,
            scene_id=scene_id,
            im_id=im_id,
            object_id=label,
            time=predictions.time.cpu().numpy(),
            detection_time=predictions.detection_time.cpu().numpy(),
            poses=poses.cpu().numpy(),
            scores=scores.cpu().numpy(),
        )

    def filter_and_save(
        self,
        predictions,
        test_list,
        time,
        save_path,
        keep_only_testing_instances=True,
    ):
        selected_idxs, predictions = self.attach_test_metadata(
            predictions=predictions,
            test_list=test_list,
            time=time,
            keep_only_testing_instances=keep_only_testing_instances,
        )
        self.save_batch_predictions(predictions, save_path)
        return selected_idxs, predictions

    def predict_batch(
        self,
        batch,
        dataset_name,
        sort_pred_by_inliers=True,
    ):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # prepare template data
        if dataset_name not in self.template_datas:
            self.set_template_data(dataset_name)

        template_data = self.template_datas[dataset_name]
        pose_recovery = self.pose_recovery[dataset_name]
        times = {"neighbor_search": None, "final_step": None}

        B, C, H, W = batch.tar_img.shape
        device = batch.tar_img.device

        # if low_memory_mode, two detections are forward at a time
        list_idx_sample = []
        if self.max_num_dets_per_forward is not None:
            for start_idx in np.arange(0, B, self.max_num_dets_per_forward):
                end_idx = min(start_idx + self.max_num_dets_per_forward, B)
                idx_sample_ = torch.arange(start_idx, end_idx, device=device)
                list_idx_sample.append(idx_sample_)
        else:
            idx_sample = torch.arange(0, B, device=device)
            list_idx_sample.append(idx_sample)

        for idx_sub_batch, idx_sample in enumerate(list_idx_sample):
            # compute target features
            tar_ae_features = self.ae_net(batch.tar_img[idx_sample])
            tar_label_np = np.asarray(
                batch.infos.label[idx_sample.cpu().numpy()]
            ).astype(np.int32)
            tar_label = torch.from_numpy(tar_label_np).to(device)

            if dataset_name == "T282":
                tar_label = torch.ones_like(tar_label)
            src_ae_features = template_data.ae_features[tar_label - 1]
            src_masks = template_data.mask[tar_label - 1]

            # Step 1: Nearest neighbor search
            self.timer.tic()
            predictions_ = self.testing_metric.test(
                src_feats=src_ae_features,
                tar_feat=tar_ae_features,
                src_masks=src_masks,
                tar_mask=batch.tar_mask[idx_sample],
                max_batch_size=None,
            )
            predictions_.infos = batch.infos.iloc[idx_sample.cpu().numpy()].reset_index(
                drop=True
            )
            if idx_sub_batch == 0:
                predictions = predictions_
            else:
                predictions.cat_df(predictions_)

        # Step 2: Find affine transforms
        num_patches = predictions.src_pts.shape[2]
        k = self.testing_metric.k
        pred_scales = torch.zeros(B, k, num_patches, device=device)
        pred_cosSin_inplanes = torch.zeros(B, k, num_patches, 2, device=device)

        self.timer.tic()
        for idx_k in range(k):
            idx_sample = torch.arange(0, B, device=device)
            idx_views = (idx_sample, predictions.id_src[:, idx_k])

            tar_label_np = np.asarray(batch.infos.label).astype(np.int32)
            tar_label = torch.from_numpy(tar_label_np).to(device)

            if dataset_name == "T282":
                tar_label = torch.ones_like(tar_label)
            src_ist_features = template_data.ist_features[tar_label - 1]
            tar_ist_features = self.ist_net.forward_by_chunk(batch.tar_img[idx_sample])

            if self.max_num_dets_per_forward is not None:
                (
                    pred_scales[:, idx_k],
                    pred_cosSin_inplanes[:, idx_k],
                ) = self.ist_net.inference_by_chunk(
                    src_feat=src_ist_features[idx_views],
                    tar_feat=tar_ist_features,
                    src_pts=predictions.src_pts[:, idx_k],
                    tar_pts=predictions.tar_pts[:, idx_k],
                    max_batch_size=self.max_num_dets_per_forward,
                )
            else:
                (
                    pred_scales[:, idx_k],
                    pred_cosSin_inplanes[:, idx_k],
                ) = self.ist_net.inference(
                    src_feat=src_ist_features[idx_views],
                    tar_feat=tar_ist_features,
                    src_pts=predictions.src_pts[:, idx_k],
                    tar_pts=predictions.tar_pts[:, idx_k],
                )

        predictions.register_tensor("relScale", pred_scales)
        predictions.register_tensor(
            "relInplane",
            pred_cosSin_inplanes,
        )
        times["neighbor_search"] = self.timer.toc()
        self.timer.reset()

        self.timer.tic()
        predictions = pose_recovery.forward_ransac(predictions=predictions)
        # sort the predictions by the number of inliers for each detection
        score = torch.sum(predictions.ransac_scores, dim=2) / num_patches
        predictions.register_tensor("scores", score)
        if sort_pred_by_inliers:
            order = torch.argsort(score, dim=1, descending=True)
            for k, v in predictions._tensors.items():
                if k in ["infos", "meta"]:
                    continue
                predictions.register_tensor(k, v[idx_sample[:, None], order])

        # calculate prediction
        pred_poses = self.pose_recovery[dataset_name].forward_recovery(
            tar_label=tar_label,
            tar_K=batch.tar_K,
            tar_M=batch.tar_M,
            pred_src_views=predictions.id_src,
            pred_M=predictions.M.clone(),
        )
        predictions.register_tensor("pred_poses", pred_poses)

        times["final_step"] = self.timer.toc()
        self.timer.reset()
        total_time = sum(times.values())
        return predictions, total_time

    def eval_retrieval(
        self,
        batch,
        idx_batch,
        dataset_name,
        sort_pred_by_inliers=True,
    ):
        predictions, total_time = self.predict_batch(
            batch=batch,
            dataset_name=dataset_name,
            sort_pred_by_inliers=sort_pred_by_inliers,
        )
        save_path = osp.join(self.log_dir, "predictions", f"{idx_batch}.npz")
        selected_idxs, predictions = self.filter_and_save(
            predictions, test_list=batch.test_list, time=total_time, save_path=save_path
        )
        return selected_idxs, predictions
    @torch.inference_mode()
    def test_step(self, batch, idx_batch):
        self.eval_retrieval(
            batch,
            idx_batch=idx_batch,
            dataset_name=self.test_dataset_name,
        )
        return 0

    def on_test_epoch_end(self):
        prediction_dir = osp.join(self.log_dir, "predictions")
        save_predictions_from_batched_predictions(
            prediction_dir,
            dataset_name=self.test_dataset_name,
            model_name=self.model_name,
            run_id=self.run_id,
        )
