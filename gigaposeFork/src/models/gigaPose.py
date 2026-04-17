import os
import os.path as osp
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
        **kwargs,
    ):
        # define the network
        super().__init__()
        self.model_name = model_name
        self.ae_net = ae_net
        self.ist_net = ist_net
        self.testing_metric = testing_metric

        self.max_num_dets_per_forward = max_num_dets_per_forward

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
        logger.info("Initializing template data ...")
        self.timer.tic()
        template_dataset = self.template_datasets[dataset_name]
        names = ["rgb", "mask", "K", "M", "poses", "ae_features", "ist_features"]
        template_data = {name: BatchedData(None) for name in names}

        for idx in tqdm(range(len(template_dataset))):
            for name in names:
                if name in ["ae_features", "ist_features"]:
                    continue
                if name == "rgb":
                    templates = template_dataset[idx].rgb.to(self.device)
                    if self.max_num_dets_per_forward is None:
                        template_data[name].append(templates)

                    ae_features = self.ae_net(templates)
                    template_data["ae_features"].append(ae_features)

                    ist_features = self.ist_net.forward_by_chunk(templates)
                    template_data["ist_features"].append(ist_features)
                else:
                    tmp = getattr(template_dataset[idx], name)
                    template_data[name].append(tmp.to(self.device))
        if self.max_num_dets_per_forward is not None:
            names.remove("rgb")
        for name in names:
            template_data[name].stack()
            template_data[name] = template_data[name].data

        self.template_datas[dataset_name] = tc.PandasTensorCollection(
            infos=pd.DataFrame(), **template_data
        )
        self.pose_recovery[dataset_name] = ObjectPoseRecovery(
            template_K=template_data["K"],
            template_Ms=template_data["M"],
            template_poses=template_data["poses"],
        )
        num_obj = len(template_data["K"])
        onboarding_time = self.timer.toc() / num_obj
        self.timer.reset()
        logger.info(f"Init {dataset_name} done! Avg time={onboarding_time} s/object")

    def filter_and_save(
        self,
        predictions,
        test_list,
        time,
        save_path,
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
        predictions = predictions[selected_idxs]

        detection_times = np.array(detection_times)
        detection_times = torch.from_numpy(detection_times).to(
            predictions.scores.device
        )
        time = torch.ones_like(detection_times) * time
        predictions.register_tensor("detection_time", detection_times)
        predictions.register_tensor("time", time)

        scene_id = np.asarray(predictions.infos.scene_id).astype(np.int32)
        im_id = np.asarray(predictions.infos.view_id).astype(np.int32)
        label = np.asarray(predictions.infos.label).astype(np.int32)

        np.savez(
            save_path,
            scene_id=scene_id,
            im_id=im_id,
            object_id=label,
            time=predictions.time.cpu().numpy(),
            detection_time=predictions.detection_time.cpu().numpy(),
            poses=predictions.pred_poses.cpu().numpy(),
            scores=predictions.scores.cpu().numpy(),
        )
        return selected_idxs, predictions

    def eval_retrieval(
        self,
        batch,
        idx_batch,
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
            idx_views = [idx_sample, predictions.id_src[:, idx_k]]

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

        save_path = osp.join(self.log_dir, "predictions", f"{idx_batch}.npz")
        selected_idxs, predictions = self.filter_and_save(
            predictions, test_list=batch.test_list, time=total_time, save_path=save_path
        )
    @torch.no_grad()
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
