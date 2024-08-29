import torch
import torch.nn as nn

class ComputeLoss:
    def __init__(self, model):
        self.device = next(model.parameters()).device
        h = model.hyp  # hyperparameters
        self.no = model.nc  # number of classes
        self.stride = model.stride
        self.reg_max = 7.5  # Example value, adjust according to your requirements
        self.BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h["cls_pw"]], device=self.device), 
            reduction='none'
        )
        self.bbox_loss = nn.SmoothL1Loss()  # Example bbox loss
        self.dfl_loss = nn.CrossEntropyLoss()  # Example dfl loss

    def __call__(self, p, targets, img=None, epoch=0):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = p[1] if isinstance(p, tuple) else p

        # Process feats to ensure tensor dimensions match
        feats = self.process_feats(feats)

        if isinstance(feats, list) and all(isinstance(xi, torch.Tensor) for xi in feats):
            cat_feats = torch.cat([xi.view(xi.shape[0], self.no, -1) for xi in feats], dim=2)
            split_size = cat_feats.shape[-1] // 2

            # Safely split the concatenated features
            pred_distri, pred_scores = self.safe_split(cat_feats, split_size)

            pred_scores = pred_scores.permute(0, 2, 1).contiguous()
            pred_distri = pred_distri.permute(0, 2, 1).contiguous()

            dtype = pred_scores.dtype
            batch_size, grid_size = pred_scores.shape[:2]
            imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
            anchor_points, stride_tensor = self.make_anchors(feats, self.stride, 0.5)

            # Preprocess targets
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])

            # Adjust targets splitting based on dimension
            gt_labels, gt_bboxes = self.split_targets(targets)

            # Safely check the shape of gt_bboxes
            mask_gt = self.get_mask_gt(gt_bboxes)

            # Decode predicted bounding boxes
            pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

            # Assignment logic
            target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt
            )

            target_bboxes /= stride_tensor
            target_scores_sum = max(target_scores.sum(), 1)

            # Compute classification loss
            loss[1] = self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

            # Compute bounding box and DFL loss if fg_mask is not empty
            if fg_mask.sum():
                loss[0], loss[2], iou = self.bbox_and_dfl_loss(
                    pred_distri,
                    pred_bboxes,
                    anchor_points,
                    target_bboxes,
                    target_scores,
                    target_scores_sum,
                    fg_mask
                )

            # Apply gains to the loss components
            loss[0] *= 7.5  # box gain
            loss[1] *= 0.5  # cls gain
            loss[2] *= 1.5  # dfl gain

        else:
            raise TypeError("Elements in feats should be tensors.")

        return loss.sum() * batch_size, loss.detach()

    def process_feats(self, feats):
        if isinstance(feats, torch.Tensor):
            return [feats]
        elif isinstance(feats, list):
            return [x for xi in feats for x in (xi if isinstance(xi, list) else [xi]) if isinstance(x, torch.Tensor)]
        else:
            raise TypeError(f"feats should be a tensor or a list of tensors: {type(feats)}")

    def make_anchors(self, feats, stride, anchor_thresh):
        # Adjust stride directly within this function
        if len(feats) != len(stride):
            stride = self._adjust_stride(feats, stride)

        anchor_points = []
        stride_tensor = torch.tensor(stride, device=self.device)
        for i, feat in enumerate(feats):
            grid_size = feat.shape[-2:]
            grid_x, grid_y = torch.meshgrid(
                torch.arange(grid_size[0], device=self.device),
                torch.arange(grid_size[1], device=self.device),
                indexing='ij'
            )
            anchors = torch.stack((grid_x, grid_y), 2).float().view(-1, 2)
            anchor_points.append(anchors)
        return anchor_points, stride_tensor

    def _adjust_stride(self, feats, stride):
        if len(feats) > len(stride):
            stride = torch.tensor(stride, device=self.device)
            stride = stride.repeat((len(feats) // len(stride)) + 1)[:len(feats)]
        else:
            stride = torch.tensor(stride, device=self.device)[:len(feats)]
        return stride

    def bbox_decode(self, anchor_points, pred_distri):
        pred_bboxes = []
        for anchors, pred in zip(anchor_points, pred_distri):
            batch_size = pred.size(0)
            num_anchors = anchors.size(0)
            num_elements = pred.numel()

            # Calculate channels
            channels = pred.size(1) // num_anchors
            if channels <= 0 or num_elements % (batch_size * num_anchors) != 0:
                raise ValueError(f"Cannot reshape pred with {num_elements} elements to match anchors with {num_anchors} anchors and batch size {batch_size}. Channels: {channels}")

            # Reshape prediction tensor to match anchors
            pred_reshaped = pred.view(batch_size, num_anchors, channels)

            # Decode bounding boxes
            decoded_bboxes = anchors + torch.sigmoid(pred_reshaped)
            pred_bboxes.append(decoded_bboxes)

        return torch.cat(pred_bboxes, dim=1)

    def preprocess(self, targets, batch_size, scale_tensor):
        targets = targets.float()
        print("Shape of targets before processing:", targets.shape)
        print("Shape of scale_tensor:", scale_tensor.shape)

        # Adjust scale_tensor to match the target shape for broadcasting
        scale_tensor = scale_tensor.view(1, -1).expand(targets.size(0), 4)
        targets[:, :4] *= scale_tensor
        print("Shape of targets after scaling:", targets.shape)
        return targets

    def split_targets(self, targets):
        if targets.dim() == 2:
            return targets[:, :1], targets[:, 1:]  # cls, xyxy
        elif targets.dim() == 3:
            return targets.split((1, 4), dim=2)  # cls, xyxy
        else:
            raise ValueError(f"Unexpected dimension for targets: {targets.dim()}")

    def get_mask_gt(self, gt_bboxes):
        if gt_bboxes.dim() < 3:
            gt_bboxes = gt_bboxes.unsqueeze(1)
        print(f"Shape of gt_bboxes: {gt_bboxes.shape}")
        return gt_bboxes.sum(dim=2, keepdim=True).gt_(0)

    def assigner(self, pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        target_labels = torch.zeros_like(pred_scores)
        target_bboxes = torch.zeros_like(pred_bboxes)
        target_scores = torch.zeros_like(pred_scores)
        fg_mask = torch.zeros_like(pred_scores[:, 0, :], dtype=torch.bool)

        for b in range(pred_scores.size(0)):
            gt_bboxes_b = gt_bboxes[b]
            # Further processing needed for the actual assignment logic

        return target_labels, target_bboxes, target_scores, fg_mask

    def safe_split(self, tensor, split_size):
        split_result = tensor.split(split_size, dim=2)
        if len(split_result) != 2:
            raise ValueError(f"Expected to split into 2 tensors but got {len(split_result)}.")
        return split_result

    def bbox_and_dfl_loss(self, pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # Compute bbox and DFL loss
        bbox_loss = self.bbox_loss(pred_distri, target_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
        dfl_loss = self.dfl_loss(pred_distri, target_bboxes)  # Placeholder for actual DFL loss computation
        return bbox_loss, dfl_loss, None  # Replace None with actual IOU computation if needed

