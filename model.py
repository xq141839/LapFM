import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.utils.transforms import SAM2Transforms
from typing import List, Tuple, Type, Optional
import os
from tqdm import tqdm
from functools import partial


class LapFM(nn.Module):
    def __init__(self, sam_model = SAM2Base, dim=24, img_size=1024):
        super(LapFM, self).__init__()

        self.device = "cuda"
        self._transforms = SAM2Transforms(
            resolution=img_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        self.model = sam_model
        self._features = None
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=(1024, 1024) # fixed
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=(1024, 1024)
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        return_ori_size: bool = True,
        img_idx: int = -1,
        p_cls: int = 0,
        c_cls: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        # if not self._is_image_set:
        #     raise RuntimeError(
        #         "An image must be set with .set_image(...) before mask prediction."
        #     )

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
            image_embedding=self._features["image_embed"][img_idx].unsqueeze(0),
            c_cls=c_cls,
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]
        low_res_masks, iou_pred, cosist = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
            p_cls=p_cls,
            c_cls=c_cls,
        )

        if return_ori_size:
            masks = F.interpolate(low_res_masks, (1024, 1024), mode="bilinear", align_corners=False)
        else:
            masks = low_res_masks
        return masks, iou_pred, cosist



  
    def forward(self, x, p_cls=1, c_cls=1):

        batch_size = x.shape[0]

        backbone_out = self.model.forward_image(x, p_cls=p_cls, c_cls=c_cls)

        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)

        vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]

        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        num_images = len(self._features["image_embed"])
        all_ious = []
        outputs_mask = []
        p_kl = []
        c_kl = []
        normalize_coords = True
        return_logits = False
        multimask_output = False
        for img_idx in range(num_images):
            # Transform input prompts
            point_coords = None
            point_labels = None
            box = None
            mask_input = None
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )

            if c_cls != 0:
                masks, iou_pred, _ = self._predict(
                    unnorm_coords,
                    labels,
                    unnorm_box,
                    mask_input,
                    multimask_output,
                    return_logits=return_logits,
                    return_ori_size = False,
                    img_idx=img_idx,
                    p_cls=p_cls,
                    c_cls=0,
                )
            else:
                masks, iou_pred, p_spital = self._predict(
                    unnorm_coords,
                    labels,
                    unnorm_box,
                    mask_input,
                    multimask_output,
                    return_logits=return_logits,
                    return_ori_size = True,
                    img_idx=img_idx,
                    p_cls=p_cls,
                    c_cls=0,
                )

            if c_cls != 0:
                masks_final, iou_pred_final, c_spital = self._predict(
                    unnorm_coords,
                    labels,
                    unnorm_box,
                    masks,
                    multimask_output,
                    return_logits=return_logits,
                    img_idx=img_idx,
                    p_cls=0,
                    c_cls=c_cls,
                )

                outputs_mask.append(masks_final.squeeze(0))
                all_ious.append(iou_pred_final.squeeze(0))
                p_kl.append(p_spital.squeeze(0))
                c_kl.append(c_spital.squeeze(0))
                
            else:

                outputs_mask.append(masks.squeeze(0))
                all_ious.append(iou_pred.squeeze(0))

        if c_cls != 0:
            return torch.stack(outputs_mask, dim=0), torch.stack(all_ious, dim=0), torch.stack(p_kl, dim=0), torch.stack(c_kl, dim=0)
        else:
            return torch.stack(outputs_mask, dim=0), torch.stack(all_ious, dim=0)










