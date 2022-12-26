from typing import Dict, List, Optional

import numpy as np
import detectron2.data.transforms as T
import torch
from torchvision.ops import nms
from detectron2.config import get_cfg
from detectron2.modeling import GeneralizedRCNN, build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Instances, Boxes, pairwise_iou

from .track import Track
from ..data.constants import INVERSE_LABEL_MAP


class TrackIDGenerator:
    def __init__(self):
        self._next_id = 0

    def __call__(self) -> int:
        res = self._next_id
        self._next_id += 1
        return res


class TracktorRCNNWrapper:
    """A wrapper around a Detectron2's GeneralizedRCNN model
    that provides an interface to both detect boxes on a new image
    and propagate given ones by passing them through the roi heads.

    TODO: refactor this and make it inherit from GeneralizedRCNN

    Usage:
        wrapper = TracktorRCNNWrapper(cfg)
        image: np.ndarray = ...
        prev_detected_instances: Instances = ...
        instances, propagated_instances = wrapper(image, prev_detected_instances)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        self.model: GeneralizedRCNN = build_model(cfg)
        if not isinstance(self.model, GeneralizedRCNN):
            raise ValueError("Model must be of type GeneralizedRCNN")
        self.model.eval()

        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)

        self.resize_transform_generator = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    def __call__(
        self,
        original_image: np.ndarray,
        detected_instances: Optional[Instances] = None,
    ) -> Dict:
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C), channels are in BGR order.
            boxes:
        """
        height, width = original_image.shape[:2]
        if (
            detected_instances is not None
            and (height, width) != detected_instances.image_size
        ):
            raise ValueError(
                "image_size of detected instances must match the dims of the passed image"
            )
        resize_transform = self.resize_transform_generator.get_transform(original_image)
        image = resize_transform.apply_image(original_image)

        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            image = image[:, :, ::-1]

        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        batched_inputs = [{"image": image, "height": height, "width": width}]
        preproc_images = self.model.preprocess_image(batched_inputs)

        with torch.no_grad():
            features = self.model.backbone(preproc_images.tensor)

            # first detect new boxes
            proposals = self.model.proposal_generator(preproc_images, features, None)[0]
            raw_detected_instances, _ = self.model.roi_heads(
                preproc_images, features, proposals, None
            )
            new_detected_instances = detector_postprocess(
                raw_detected_instances[0], height, width
            )

            result = {"instances": new_detected_instances}

            if detected_instances is not None:
                # then regress the given ones
                if len(detected_instances) == 0:
                    result["propagated_instances"] = (
                        detected_instances,
                        torch.Tensor().to(self.model.device),
                    )
                else:
                    # first we need to resize them to the model input resolution
                    # and them wrap them into an Instances object
                    if isinstance(resize_transform, T.NoOpTransform):
                        resized_dims = (height, width)
                        scale_x, scale_y = 1, 1
                    else:
                        resized_dims = (resize_transform.new_h, resize_transform.new_w)
                        scale_x, scale_y = (
                            resized_dims[1] / width,
                            resized_dims[0] / height,
                        )

                    di_boxes = detected_instances.pred_boxes.clone()
                    di_boxes.scale(scale_x, scale_y)
                    detected_instances_as_proposals = Instances(
                        resized_dims, proposal_boxes=di_boxes
                    )

                    # NOTE: we need to run box regression manually to get the indices of the boxes
                    # which are kept during the box predictor NMS

                    box_features = self.model.roi_heads.box_pooler(
                        [features[f] for f in self.model.roi_heads.box_in_features],
                        [detected_instances_as_proposals.proposal_boxes],
                    )
                    box_features = self.model.roi_heads.box_head(box_features)
                    predictions = self.model.roi_heads.box_predictor(box_features)
                    del box_features

                    (
                        raw_regressed_instances,
                        nms_kept_instances_idxs,
                    ) = self.model.roi_heads.box_predictor.inference(
                        predictions, [detected_instances_as_proposals]
                    )

                    regressed_instances = detector_postprocess(
                        raw_regressed_instances[0], height, width
                    )
                    result["propagated_instances"] = (
                        regressed_instances,
                        nms_kept_instances_idxs[0],
                    )

            del features
        return result


class Tracktor:
    def __init__(
        self,
        object_detector_config: str,
        object_detector_weights: str,
        sigma_active: float = 0.5,
        active_tracks_nms_thresh: float = 0.6,
        new_tracks_nms_thresh: float = 0.3,
        inactive_patience: int = 1,
        max_active_tracks: int = 20,
        ignore_balls: bool = True,
    ):
        # Setup base object detector config
        self.object_detector_cfg = get_cfg()
        self.object_detector_cfg.merge_from_file(object_detector_config)
        self.object_detector_cfg.MODEL.WEIGHTS = object_detector_weights

        self.sigma_active = sigma_active
        self.active_tracks_nms_thresh = active_tracks_nms_thresh
        self.new_tracks_nms_thresh = new_tracks_nms_thresh

        self.inactive_patience = inactive_patience
        self.max_active_tracks = max_active_tracks

        self.ignore_balls = ignore_balls
        if not self.ignore_balls:
            raise NotImplementedError("Ball tracking not supported!")

        self._object_detector = TracktorRCNNWrapper(self.object_detector_cfg)

        self.reset()

    @classmethod
    def from_config(cls, cfg):
        tracker_cfg = cfg.TRACKER
        return cls(
            object_detector_config=tracker_cfg.OBJECT_DETECTOR.CONFIG,
            object_detector_weights=tracker_cfg.OBJECT_DETECTOR.WEIGHTS,
            sigma_active=tracker_cfg.SIGMA_ACTIVE,
            active_tracks_nms_thresh=tracker_cfg.ACTIVE_TRACKS_NMS_THRESH,
            new_tracks_nms_thresh=tracker_cfg.NEW_TRACKS_NMS_THRESH,
            inactive_patience=tracker_cfg.INACTIVE_PATIENCE,
            max_active_tracks=tracker_cfg.MAX_ACTIVE_TRACKS,
            ignore_balls=tracker_cfg.IGNORE_BALLS,
        )

    def reset(self):
        self._track_id_generator = TrackIDGenerator()
        self._tracks: List[Track] = []
        self._inactive_tracks: List[Track] = []
        self._current_frame_id = None
        self._prev_frame_id = None

    def get_active_tracks(self):
        return self._tracks

    def get_inactive_tracks(self):
        return self._inactive_tracks

    def get_active_tracks_positions(self):
        return [t.current_pos for t in self._tracks]

    def get_all_tracks(self):
        """
        Returns both active and inactive tracks sorted by the starting frame id.
        """
        return sorted(
            self._tracks + self._inactive_tracks, key=lambda x: x.start_frame_id
        )

    def track(self, frame_id: int, image: np.ndarray):

        active_tracks = []
        deactivated_tracks = []

        # Should we check that these are always increasing?
        self._last_frame_id = self._current_frame_id
        self._current_frame_id = frame_id

        if self._tracks is None or len(self._tracks) == 0:
            res = self._object_detector(image)
            new_instances = res["instances"]

            if self.ignore_balls:
                new_instances = new_instances[
                    torch.ne(new_instances.pred_classes, INVERSE_LABEL_MAP["ball"])
                ]
        else:
            height, width = image.shape[:2]
            # Wrap current tracks positions into a d2 Boxes object and move to
            # target device then wrap into a d2 Instances object
            propagated_instances = Instances(
                image_size=(height, width),
                pred_boxes=Boxes(
                    torch.from_numpy(
                        np.array(
                            [t.current_pos for t in self._tracks], dtype=np.float32
                        )
                    )
                ).to(torch.device(self.object_detector_cfg.MODEL.DEVICE)),
            )

            # Propagate existing tracks via bbox regression and detect new instances
            res = self._object_detector(image, propagated_instances)
            new_instances = res["instances"]
            regressed_instances, kept_instances_idxs = res["propagated_instances"]

            if self.ignore_balls:
                new_instances = new_instances[
                    torch.ne(new_instances.pred_classes, INVERSE_LABEL_MAP["ball"])
                ]

            # Filter out invalid tracks
            regressed_instances_valid_mask = torch.logical_and(
                regressed_instances.pred_boxes.nonempty(),
                torch.ge(regressed_instances.scores, self.sigma_active),
            )
            regressed_instances = regressed_instances[regressed_instances_valid_mask]
            kept_instances_idxs = kept_instances_idxs[regressed_instances_valid_mask]

            # Apply nms between valid propagated instances and new instances (if any)
            if len(new_instances) > 0:
                # Apply nms between valid_track and the new instances
                # The following loop was adapted from
                # https://github.com/phil-bergmann/tracking_wo_bnw/blob/master/src/tracktor/tracker.py#L336
                keep = torch.arange(len(new_instances)).to(new_instances.scores.device)
                for i in range(len(regressed_instances)):
                    nms_boxes = torch.cat(
                        [
                            regressed_instances.pred_boxes.tensor[[i]],
                            new_instances.pred_boxes.tensor,
                        ]
                    )
                    nms_scores = torch.cat(
                        [
                            torch.Tensor([2.0]).to(new_instances.scores.device),
                            new_instances.scores,
                        ]
                    )
                    keep = nms(nms_boxes, nms_scores, self.active_tracks_nms_thresh)
                    keep = keep[torch.ge(keep, 1)] - 1
                    new_instances = new_instances[keep]
                    if keep.nelement() == 0:
                        break

            if len(new_instances) > 0:
                # Finally make sure the iou of new instances with any of
                # the existing tracks is lower than the threshold
                new_regressed_instances_max_iou, _ = torch.max(
                    pairwise_iou(
                        new_instances.pred_boxes, regressed_instances.pred_boxes
                    ),
                    1,
                )
                new_instances = new_instances[
                    torch.lt(
                        new_regressed_instances_max_iou, self.new_tracks_nms_thresh
                    )
                ]

            # Convert regressed tracks back to numpy and update tracks
            kept_instances_idxs = kept_instances_idxs.detach().cpu().numpy()
            regressed_instances_boxes = (
                regressed_instances.pred_boxes.tensor.detach().cpu().numpy()
            )
            for regressed_instance_idx, track_idx in enumerate(kept_instances_idxs):
                self._tracks[track_idx].update(
                    self._current_frame_id,
                    regressed_instances_boxes[regressed_instance_idx],
                )
            lost_tracks_idxs = np.setdiff1d(
                np.arange(len(self._tracks)), kept_instances_idxs
            )
            for track_idx in lost_tracks_idxs:
                self._tracks[track_idx].update(self._current_frame_id, None)

            for track in self._tracks:
                if track.inactive_count > self.inactive_patience:
                    deactivated_tracks.append(track)
                else:
                    active_tracks.append(track)

        if len(new_instances) > 0:
            #  initialize new tracks
            for box, class_id in zip(
                new_instances.pred_boxes.tensor.detach().cpu().numpy(),
                new_instances.pred_classes.detach().cpu().numpy(),
            ):
                if len(active_tracks) >= self.max_active_tracks:
                    break
                new_track = Track(
                    track_id=self._track_id_generator(),
                    class_id=class_id,
                    start_frame_id=self._current_frame_id,
                    position=box,
                )
                active_tracks.append(new_track)

        self._tracks = active_tracks
        self._inactive_tracks += deactivated_tracks

        active_tracks_info = []
        for t in sorted(self._tracks, key=lambda x: x.track_id):
            active_tracks_info.append(
                {
                    "frame_id": t.last_frame_id,
                    "track_id": t.track_id,
                    "bbox": t.current_pos,
                }
            )
        return active_tracks_info
