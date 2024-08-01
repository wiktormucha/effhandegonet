from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import numpy as np
import os
import torchmetrics as metrics
import matplotlib.pyplot as plt
import copy as copy
import torchvision.transforms as T
from utils.testing import (
    batch_epe_calculation,
    batch_auc_calculation,
    batch_pck_calculation,
)
from datasets.h2o import H2O_Dataset_hand_train
import pandas as pd
from tqdm import tqdm
from spock_dataclasses import Data, TrainingConfig
from utils.general_utils import (
    EgocentricModelType,
    make_model,
    draw_keypoints_on_single_hand
)
from utils.egocentric import (
    get_egocentric_prediction_mediapipe,
    save_predictions,
    get_egocentric_predictions_effhandegonet,
    get_egocentric_predictions_effhandnet,
    get_egocentric_predictions_poseresnet50
)
MAX_NUM_THREADS = 1
torch.set_num_threads(MAX_NUM_THREADS)
SAVE_PREDICTED_POSE = False
# SAVE_GT_IMAGE = True

# MODEL_TYPE = EgocentricModelType.mediapipe
MODEL_TYPE = EgocentricModelType.effhandegonet
# MODEL_TYPE = EgocentricModelType.effhandnet
# MODEL_TYPE = EgocentricModelType.poseresnet50
DEVICE = 7

config = {
    "device": DEVICE,
}
# "model_path": "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/custom_heatmap_allaug5_61",
if MODEL_TYPE == EgocentricModelType.poseresnet50:

    from mmpose.apis import init_model
    from mmpose.utils import register_all_modules
    results_file_path = "model_poseresnet50/test_asdf.csv"
    factor = 15
    yolo_labels_dir = "/caa/Homes01/wmucha/repos/yolov7/h2o_bb_hands_test/exp/labels"
    register_all_modules()
    config_file = "/caa/Homes01/wmucha/repos/mmpose_test/mmpose-dev-1.x/configs/hand_2d_keypoint/topdown_heatmap/freihand2d/td-hm_res50_8xb64-100e_freihand2d-224x224.py"
    checkpoint_file = "/caa/Homes01/wmucha/repos/mmpose_test/mmpose-dev-1.x/configs/hand_2d_keypoint/topdown_heatmap/freihand2d/res50_freihand_224x224-ff0799bc_20200914.pth"
    model = init_model(config_file, checkpoint_file,
                       device=config['device'])  # or device='cuda:0'
    folder_to_save = "hand_pose_resnet50"
    model_cfg = Data(
        path="/data/wmucha/datasets",
        batch_size=1,
        img_size=(1280, 720),
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
    )
    train_cfg = TrainingConfig(
        device=DEVICE,
        lr=0.1,
        max_epochs=1,
        weight_decay=0.0,
        momentum=0.0,
        grad_clipping=0.0,
        load_model=True,
        load_model_path='/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/custom_heatmap_allaug5_61',
        model_type='CustomHeatmapsModel',
        run_name='asdf',
        train_flag=False,
        test_flag=True,
        one_batch_flag=False,
        experiment_name='asdf',
        early_stopping=0,
    )
    albumentation_val = A.Compose(
        [
            A.Resize(model_cfg.img_size[1], model_cfg.img_size[0]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

if MODEL_TYPE == EgocentricModelType.effhandnet:
    yolo_labels_dir = "/caa/Homes01/wmucha/repos/yolov7/h2o_bb_hands_test/exp/labels"
    factor = 15
    model_cfg = Data(
        path="/data/wmucha/datasets",
        batch_size=1,
        img_size=(1280, 720),
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
    )

    results_file_path = "model_effhandnet/test_asdf.csv"
    folder_to_save = "hand_pose_ownmodel"
    # make folder to save the results if does not exist
    if not os.path.exists("model_effhandnet"):
        os.makedirs("model_effhandnet")

    albumentation_val = A.Compose(
        [
            A.Resize(model_cfg.img_size[1], model_cfg.img_size[0]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    train_cfg = TrainingConfig(
        device=DEVICE,
        lr=0.1,
        max_epochs=1,
        weight_decay=0.0,
        momentum=0.0,
        grad_clipping=0.0,
        load_model=True,
        load_model_path='saved_models/EffHandNet_FreiHAND_128x128.pth',
        model_type='CustomHeatmapsModel',
        run_name='asdf',
        train_flag=False,
        test_flag=True,
        one_batch_flag=False,
        experiment_name='asdf',
        early_stopping=0,
    )
    model = make_model(model_cfg=train_cfg, device=train_cfg.device)
    model.eval()


if MODEL_TYPE == EgocentricModelType.mediapipe:
    import mediapipe as mp
    model_cfg = Data(
        path="/data/wmucha/datasets",
        batch_size=1,
        img_size=(1280, 720),
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
    )
    results_file_path = "model_media/test_asdf.csv"
    folder_to_save = "hand_pose_mediapipe"

    if not os.path.exists("model_media"):
        os.makedirs("model_media")

    albumentation_val = A.Compose(
        [
            A.Resize(model_cfg.img_size[1], model_cfg.img_size[0]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.2
    )
    model = mp.solutions.hands.Hands()

elif MODEL_TYPE == EgocentricModelType.effhandegonet:
    results_file_path = "model_egocentric/nicnicnic.csv"

    if not os.path.exists("model_egocentric"):
        os.makedirs("model_egocentric")
    folder_to_save = "hand_pose_ownmodel_ego"
    model_cfg = Data(
        path="/data/wmucha/datasets",
        batch_size=1,
        img_size=[512, 512],
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
    )
    activation = torch.nn.Softmax(dim=1)
    albumentation_val = A.Compose(
        [
            A.Resize(model_cfg.img_size[1], model_cfg.img_size[0]),
            A.Normalize(mean=model_cfg.norm_mean,
                        std=model_cfg.norm_std),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )
    train_cfg = TrainingConfig(
        device=DEVICE,
        lr=0.1,
        max_epochs=1,
        weight_decay=0.0,
        momentum=0.0,
        grad_clipping=0.0,
        load_model=True,
        load_model_path='saved_models/EffHandEgoNet_H2O_512x512.pth',
        model_type='CustomEgocentric',
        run_name='asdf',
        train_flag=False,
        test_flag=True,
        one_batch_flag=False,
        experiment_name='asdf',
        early_stopping=0,
        num_workers=12,
        criterion='SGD',
        data_type='2D',
        dataset='h2o',

    )
    model = make_model(model_cfg=train_cfg, device=train_cfg.device)
    model.eval()


test_dataset = H2O_Dataset_hand_train(
    config=model_cfg, type="test", albu_transform=albumentation_val
)

test_dataloader = DataLoader(
    test_dataset,
    model_cfg.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=6,
    pin_memory=True,
)


if os.path.exists(results_file_path):
    df = pd.read_csv(results_file_path)
else:
    df = pd.DataFrame(
        columns=["img", "pck_acc", "epe_lst",
                 "auc_lst", "close_hands", "far_scenario"]
    )


gt_hand_left = []
gt_hand_right = []
pred_hand_left = []
pred_hand_right = []
far_scenario = False
close_scenario = False

close_scenario_paths = []
far_scenario_paths = []

epe_lst = []
pck_lst = []
auc_lst = []


for i, data in enumerate(tqdm(test_dataloader, 0)):
    img = data["img"].to(config["device"])
    img_path = data["img_path"]

    if isinstance(img_path, list):
        img_path = img_path[0]

    if not df.empty:
        if img_path in df["img"].values:
            continue

    pth_to_save = img_path[0:60]
    filename_to_save = img_path[65:].replace(".png", ".txt")
    dirname = os.path.join(pth_to_save, folder_to_save)

    if os.path.isdir(dirname) == False:
        os.mkdir(dirname)

    gt_ptsL = data["keypoints_left"].cpu().detach().numpy() * (1280, 720)
    gt_ptsR = data["keypoints_right"].cpu().detach().numpy() * (1280, 720)

    gt_hand_left.append(data["left_hand_flag"])
    gt_hand_right.append(data["right_hand_flag"])

    # Check the value of "MODEL_TYPE" and call the appropriate function to get the egocentric hand predictions
    if MODEL_TYPE == EgocentricModelType.mediapipe:
        pred_hand_left, pred_hand_right, pred_left_np, pred_right_np = get_egocentric_prediction_mediapipe(img=img, hand_model=model, pth_to_save=pth_to_save, folder_to_save=folder_to_save,
                                                                                                           filename_to_save=filename_to_save, pred_hand_left=pred_hand_left, pred_hand_right=pred_hand_right)
    elif MODEL_TYPE == EgocentricModelType.effhandegonet:
        pred_left_flag, pred_right_flag, pred_left_np, pred_right_np = get_egocentric_predictions_effhandegonet(
            img=img, hand_model=model, activation=activation, pred_hand_left=pred_hand_left, pred_hand_right=pred_hand_right)
    elif MODEL_TYPE == EgocentricModelType.effhandnet:
        # Read in the YOLO labels file for this image
        temp_pth = os.path.join(yolo_labels_dir, "{:06d}.txt".format(i))
        yolo_labels_file = pd.read_csv(
            temp_pth, sep=" ", header=None, index_col=None)

        # Call the function to get the egocentric hand predictions using the YOLO labels
        pred_left_flag, pred_right_flag, pred_left_np, pred_right_np = get_egocentric_predictions_effhandnet(
            img=img, hand_model=model, yolo_labels_file=yolo_labels_file, factor=factor, device=train_cfg.device, pred_hand_left=pred_hand_left, pred_hand_right=pred_hand_right)

    elif MODEL_TYPE == EgocentricModelType.poseresnet50:
        # Read in the YOLO labels file for this image
        temp_pth = os.path.join(yolo_labels_dir, "{:06d}.txt".format(i))
        yolo_labels_file = pd.read_csv(
            temp_pth, sep=" ", header=None, index_col=None)

        # Call the function to get the egocentric hand predictions using the YOLO labels
        pred_left_flag, pred_right_flag, pred_left_np, pred_right_np = get_egocentric_predictions_poseresnet50(
            img=img, hand_model=model, yolo_labels_file=yolo_labels_file, factor=factor, device=train_cfg.device, pred_hand_left=pred_hand_left, pred_hand_right=pred_hand_right)

    # If "SAVE_PREDICTED_POSE" is True, save the predicted poses to file
    if SAVE_PREDICTED_POSE:
        save_predictions(pred_left_np=pred_left_np,
                         pred_right_np=pred_right_np, pth_to_save=pth_to_save,)

    # Initialize some variables for calculating accuracy metrics
    final_epe = 0
    epe_count = 0
    final_pck = 0
    final_auc = 0

    # If the ground truth left hand flag is 1 and the predicted left hand flag is 1, calculate the EPE, PCK, and AUC metrics for the left hand
    if gt_hand_left[-1] == 1 and pred_hand_left[-1] == 1:
        final_epe += batch_epe_calculation(
            np.reshape(pred_left_np, (1, 21, 2)), gt_ptsL)

        final_pck += batch_pck_calculation(
            np.reshape(pred_left_np, (1, 21, 2)), gt_ptsL)

        final_auc += batch_auc_calculation(
            np.reshape(pred_left_np, (1, 21, 2)), gt_ptsL)

        epe_count += 1

    # If the ground truth right hand flag is 1 and the predicted right hand flag is 1, calculate the EPE, PCK, and AUC metrics for the right hand
    if gt_hand_right[-1] == 1 and pred_hand_right[-1] == 1:
        final_epe += batch_epe_calculation(
            np.reshape(pred_right_np, (1, 21, 2)), gt_ptsR)
        final_pck += batch_pck_calculation(
            np.reshape(pred_right_np, (1, 21, 2)), gt_ptsR)
        final_auc += batch_auc_calculation(
            np.reshape(pred_right_np, (1, 21, 2)), gt_ptsR)

        epe_count += 1

    # If both the ground truth left and right hand flags are 1 and the predicted left and right hand flags are 1, calculate the IoU metrics
    if gt_hand_left[-1] == 1 and gt_hand_right[-1] == 1 and pred_hand_left[-1] == 1 and pred_hand_right[-1] == 1:

        xmax_l, ymax_l = gt_ptsL[0].max(axis=0)
        xmin_l, ymin_l = gt_ptsL[0].min(axis=0)
        xmax_r, ymax_r = gt_ptsR[0].max(axis=0)
        xmin_r, ymin_r = gt_ptsR[0].min(axis=0)

        if xmin_r - xmax_l < -100:
            close_scenario = (True)
            close_scenario_paths.append(img_path)
            far_scenario = False
        else:
            close_scenario = (False)

            if xmin_r - xmax_l > 400:
                far_scenario = True
                far_scenario_paths.append(img_path)

    else:
        close_scenario = (False)
        far_scenario = False

    # Calculate the final EPE, PCK, and AUC metrics
    epe_count = max(1, epe_count)
    epe = (final_epe)/epe_count
    pck = (final_pck)/epe_count
    auc = (final_auc)/epe_count

    # If "False" (which it always is), save some images and data to file
    if False:
        # Concatenate "gt_ptsL" and "gt_ptsR" into a single array called "gt_keyboards"
        gt_keyboards = np.concatenate((gt_ptsL, gt_ptsR))

        if SAVE_GT_IMAGE:
            plt.clf()
            plt.imshow(img)
            draw_keypoints_on_single_hand(gt_keyboards[0])
            draw_keypoints_on_single_hand(gt_keyboards[1])
            plt.savefig("wrong_pred/gt_" + str(idx) + ".png")

        if SAVE_GT_IMAGE:
            plt.clf()
            plt.imshow(img)
            draw_keypoints_on_single_hand(pred[0])
            draw_keypoints_on_single_hand(pred[1])
            plt.savefig("wrong_pred/mediapipe_" + str(idx) + ".png")

    # df.loc[len(df.index)] = [img_path, pck, epe,
    #                          auc, close_scenario, far_scenario]
    # df.to_csv(results_file_path, index=False)
    epe_lst.append(epe)
    pck_lst.append(pck)
    auc_lst.append(auc)
    # break

# Save the ground truth and predicted hand flags to file
gt_hand_left_df = pd.DataFrame(gt_hand_left)
gt_hand_right_df = pd.DataFrame(gt_hand_right)
pred_hand_left_df = pd.DataFrame(pred_hand_left)
pred_hand_right_df = pd.DataFrame(pred_hand_right)


clos_scenarios_pd = pd.DataFrame(close_scenario_paths)
far_scenarios_pd = pd.DataFrame(far_scenario_paths)

clos_scenarios_pd.to_csv("close_scenarios.csv")
far_scenarios_pd.to_csv("far_scenarios.csv")

gt_hand_left_df.to_csv(str(MODEL_TYPE)+"_gt_hand_left_df.csv")
gt_hand_right_df.to_csv(str(MODEL_TYPE)+"_gt_hand_right_df.csv")
pred_hand_left_df.to_csv(str(MODEL_TYPE)+"_pred_hand_left_df.csv")
pred_hand_right_df.to_csv(str(MODEL_TYPE)+"_pred_hand_right_df.csv")

# Calculate the accuracy metrics for the predicted hand flags
Accuracy = metrics.Accuracy(task="binary")

acc_left = Accuracy(torch.Tensor(pred_hand_left),
                    torch.Tensor(gt_hand_left))
acc_right = Accuracy(torch.Tensor(pred_hand_right),
                     torch.Tensor(gt_hand_right))

# Print the accuracy metrics for the predicted hand flags
print("Acc left: ", acc_left, "no. ", len(gt_hand_left))
print("Acc right: ", acc_right, "no. ", len(gt_hand_right))
print("PCK: ", np.mean(pck_lst))
print("EPE: ", np.mean(epe_lst))
print("AUC: ", np.mean(auc_lst))
