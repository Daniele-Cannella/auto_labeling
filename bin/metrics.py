import torchvision
import torch
from sklearn.metrics import auc
from log import Log


class Metrics:
    def __init__(self, confidence_scores: list) -> None:
        """
        :params
            confidence_scores: lista di [tp, fp, fn] per ogni confidence
        :return:
        """
        self.confidence_scores = confidence_scores


    def get_precision_recall(self) -> tuple:
        """
        Calcola la media dell'auc per ogni confidence e per ognuna confidence calcola precision e recall
        :params
            self:
        :return
            tuple: contriene auc medio e lista di precision e recall per ogni confidence
        """
        results = []
        auc_score = 0

        for i in range(0, len(self.confidence_scores)):
            tp = self.confidence_scores[i][0]
            fp = self.confidence_scores[i][1]
            fn = self.confidence_scores[i][2]

            recall = tp / (tp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1e-6
            auc_score += auc(recall, precision)

            results.append({'confidence': i + 1 / 10, 'precision': precision, 'recall': recall, 'auc': auc_score})
        mean_auc_score = auc_score / len(self.confidence_scores)
        return (mean_auc_score, results)


def get_confMatr(predictions: list[tuple], ground_truths: list[tuple], class_id: int):
    class_prs = [torch.Tensor(bbox) for _, bbox in predictions]
    class_grs = [torch.Tensor(bbox) for idx, bbox in ground_truths if idx == class_id]

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    print("class_prs", class_prs, "class_grs", class_grs)

    class_grs = torch.stack(class_grs, dim=0)
    class_prs = torch.stack(class_prs, dim=0)
    if class_grs.numel() != 0 and class_prs.numel() != 0:
        # Convert lists of boxes to tensors
        gr_bbxs = torchvision.ops.box_convert(
            boxes=torch.Tensor(class_grs),
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )

        pr_bbxs = torchvision.ops.box_convert(
            boxes=torch.Tensor(class_prs),
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )

        # pr_bbxs = torch.stack(class_prs, dim=0)

        # Find matching pairs (if any) based on IoU
        iou_matrix = torchvision.ops.box_iou(gr_bbxs, pr_bbxs)  # Efficient IoU calculation
        matched_indices = torch.where(
            iou_matrix >= 0.5
        )  # Assuming IoU threshold of 0.5

        true_positives = matched_indices[0].unique().numel()

        false_positives = len(class_prs) - true_positives

        false_negatives = len(class_grs) - true_positives
    elif not class_prs:
        false_negatives = 1
    elif not class_grs:
        false_positives = 1

    return true_positives, false_positives, false_negatives
