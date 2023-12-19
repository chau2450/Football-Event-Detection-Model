
import torch
from collections import Counter



class yolo_utils:
    def __init__(self) -> None:
        pass
    
    
    @staticmethod
    def intersection_over_union(boxes_preds: torch.Tensor, boxes_labels: torch.Tensor, 
                                box_format: str="midpoint") -> float:
        """
        Each output tensor is in the the shape (7x7,20+5x2):
            20 class probabilities, x,y,w,h and conference score for each bounding box
            Based on the YOLO V1 paper we are assuming two bounding boxes per grid 
            (this code is not modular to the grids and bounding box preds, 
            could be an improvement for the future)

        Args:
            boxes_preds (_type_): _description_
            boxes_labels (_type_): _description_
            box_format (str, optional): _description_. Defaults to "midpoint".

        Returns:
            _type_: _description_
        """


        if box_format == "midpoint":
            box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
            box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
            box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
            box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
            box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
            box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
            box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
            box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

        elif box_format == "corners":
            box1_x1 = boxes_preds[..., 0:1]
            box1_y1 = boxes_preds[..., 1:2]
            box1_x2 = boxes_preds[..., 2:3]
            box1_y2 = boxes_preds[..., 3:4]
            box2_x1 = boxes_labels[..., 0:1]
            box2_y1 = boxes_labels[..., 1:2]
            box2_x2 = boxes_labels[..., 2:3]
            box2_y2 = boxes_labels[..., 3:4]
        
        else:
            assert "Chosen format not supported"

        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)

        # need to use clamp if intersection is 0
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

        
        # add small constant when either of the bounding boxes has zero area (/0 error)
        union = box1_area + box2_area - intersection + 1e-6

        return intersection / union
    
    
    @staticmethod
    def non_max_suppression(bounding_boxes: list, iou_thresh: float, 
                            p_thresh: float, box_format: str="midpoint")-> list:
        """_summary_

        Args:
            bounding_boxes (list): _description_
            iou_thresh (float): _description_
            p_thresh (float): _description_
            box_format (str, optional): _description_. Defaults to "midpoint".

        Returns:
            list: _description_
        """
        
        assert box_format in ["corners", "midpoint"], "Invalid box format"

        # Filter out boxes with a low detection probability
<<<<<<< HEAD
<<<<<<< HEAD
        filtered_boxes = [box for box in bounding_boxes if box[-1] > p_thresh]
=======
        filtered_boxes = [box for box in bounding_boxes if box[1] > p_thresh]
>>>>>>> 1736d8d6c4d9c2cbb6f0cb48ac2035220f998363
=======
        filtered_boxes = [box for box in bounding_boxes if box[1] > p_thresh]
>>>>>>> 1736d8d6c4d9c2cbb6f0cb48ac2035220f998363
        
        # Sort the boxes by score in descending order
        filtered_boxes.sort(key=lambda x: x[1], reverse=True)

        selected_boxes = []

        while len(filtered_boxes) >0:
            
            # first as we sorted the box
            box_chosen = filtered_boxes.pop(0)
            
            remaining_boxes = []
            
            for box in filtered_boxes:
                if box[0] != box_chosen[0] or yolo_utils.intersection_over_union(torch.tensor(box_chosen[2:]),
                                            torch.tensor(box[2:]), box_format=box_format) < iou_thresh:
                    
                    remaining_boxes.append(box)
            
            
            filtered_boxes = remaining_boxes
            
            selected_boxes.append(box_chosen)
                    
                

        return selected_boxes
    
    
    
    @staticmethod
    def mean_avg_precision(
        pred_boxes: list, ground_truth_boxes: list, box_format="midpoint", num_classes:int = 20,
        iou_threshold=0.5
    ) -> float:
        """_summary_

        Args:
            pred_boxes (list): list of bounding box predictions in the format[img_index_id,class,p_score,x1,y1,x2,y2]
            ground_truth_boxes (list): _description_
            box_format (str, optional): _description_. Defaults to "midpoint".
            num_classes (int, optional): _description_. Defaults to 20. This is because I am using the PASCAL VOC dataset

        Returns:
            float: _description_
        """
        
        avg_precisions:list = []
        
        # /0 error prevention
        eps = 1e-6
        
        # loop by class
        for c in range(num_classes):
            detections: list = []
            ground_truths: list = []
            
            # only get the pred && group truth for this class iteration
            for d in pred_boxes:
                if d[1] == c:
                    detections.append(d)
                
            for t in ground_truth_boxes:
                if t[1] == c:
                    ground_truths.append(t)
            
            # get dict of GT count per image ID
            bounding_boxes_count = Counter()
            
            for gt in ground_truths:
                
                bounding_boxes_count[gt[0]] += 1
            
            
            # For example, if bounding_boxes_count is {0: 3, 1: 2}, 
            # then after this loop, it will become {0: tensor([0., 0., 0.]), 
            # 1: tensor([0., 0.])}. This means that for image 0, 
            # there are three ground truth bounding boxes that 
            # have not yet been matched with any predictions, and for image 1, there are two.
            
            for key, val in bounding_boxes_count.items():
                bounding_boxes_count[key] = torch.zeros(val)
            
            
            # Sort the detections in descending order by their confidence score
            detections.sort(key=lambda x: x[2], reverse=True)
            
            # len of the detections as we need to classify one or the other (TP/FP)
            TP = torch.zeros((len(detections)))
            FP = torch.zeros((len(detections)))
            
            total_true_bounding_boxes = len(ground_truths)
            
            
            # loop through each BB detection
            for detection_id, detection in enumerate(detections):
                
                # only compare to the same image
                ground_truth_img = [
                        bbox for bbox in ground_truths if bbox[0] == detection[0]
                    ]
                
                # num_gts = len(ground_truth_img)
            
                best_iou = 0
                best_gt_idx = -1

                # Iterate over each ground truth box from the same image.
                for idx, gt in enumerate(ground_truth_img):
                    # Calculate the IoU between the current detection and the ground truth box.
                    iou = yolo_utils.intersection_over_union(
                        torch.tensor(detection[3:]),  # The detection bbox coordinates
                        torch.tensor(gt[3:]),         # The ground truth bbox coordinates
                        box_format=box_format,
                    )

                    # If the calculated IoU is greater than the best IoU seen so far,
                    # update the best IoU and the index of the best ground truth box.
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                # After comparing with all ground truths, check if the best IoU is above the threshold.
                if best_iou > iou_threshold:
                    # Check if this ground truth box has not been detected/matched before.
                    if bounding_boxes_count[detection[0]][best_gt_idx] == 0:
                        # Mark the detection as a True Positive (TP) and mark the ground truth box as detected.
                        TP[detection_id] = 1
                        bounding_boxes_count[detection[0]][best_gt_idx] = 1
                    else:
                        # If the ground truth box has already been detected, mark this as a False Positive (FP).
                        FP[detection_id] = 1
                else:
                    # If the IoU is below the threshold, mark the detection as a False Positive (FP).
                    FP[detection_id] = 1
 
            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_bounding_boxes + eps)
            precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + eps))
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            # torch.trapz for numerical integration
            avg_precisions.append(torch.trapz(precisions, recalls))

        return sum(avg_precisions) / len(avg_precisions)


<<<<<<< HEAD
<<<<<<< HEAD
    @staticmethod
    def decode_predictions(predictions, S=7, B=2, C=20, confidence_threshold=0.5, nms_threshold=0.4):
        """
        Decodes the predictions of a YOLO model.
        """
        boxes = []
        predictions = predictions.reshape(S, S, C + B*5)
        for i in range(S):
            for j in range(S):
                cell_predictions = predictions[i, j, :]
                for b in range(B):
                    bx = cell_predictions[C + b*5:C + (b+1)*5]
                    x_center, y_center, width, height, confidence = bx
                    if confidence > confidence_threshold:
                        x_center, y_center, width, height = yolo_utils.convert_to_absolute(
                            x_center, y_center, width, height, S, i, j
                        )
                        boxes.append([x_center, y_center, width, height, confidence])
        # Apply Non-Maximum Suppression
        boxes = yolo_utils.non_max_suppression(boxes, nms_threshold, confidence_threshold)

        return boxes

    @staticmethod
    def convert_to_absolute(x_center, y_center, width, height, grid_size, cell_row, cell_col):
        """
        Convert relative coordinates to absolute coordinates.
        """
        cell_size = 1.0 / grid_size
        x_center = (cell_col + x_center) * cell_size
        y_center = (cell_row + y_center) * cell_size
        width *= cell_size
        height *= cell_size

        # Convert to top-left and bottom-right
        x1 = int((x_center - width / 2) * 448)  # Assuming input size is 448x448
        y1 = int((y_center - height / 2) * 448)
        x2 = int((x_center + width / 2) * 448)
        y2 = int((y_center + height / 2) * 448)

        return x1, y1, x2, y2
=======
>>>>>>> 1736d8d6c4d9c2cbb6f0cb48ac2035220f998363
=======
>>>>>>> 1736d8d6c4d9c2cbb6f0cb48ac2035220f998363
