from utils.testing import batch_epe_calculation, batch_auc_calculation, batch_pck_calculation
from models.models import EffHandNet
from datasets.FreiHAND import get_FreiHAND_dataloaders
from utils.general_utils import heatmaps_to_coordinates
from tqdm import tqdm
import torch


def evaluate(model, dataloader, using_heatmaps=True, batch_size=0):
    accuracy_all = []
    pck_acc = []
    epe_lst = []
    auc_lst = []

    for data in tqdm(dataloader):
        inputs = data["img"]
        pred_heatmaps = model(inputs)
        pred_heatmaps = pred_heatmaps.detach().numpy()
        true_keypoints = (data["keypoints"]).numpy()

        if using_heatmaps == True:
            pred_keypoints = heatmaps_to_coordinates(
                pred_heatmaps, img_size=128)
        else:
            pred_keypoints = pred_heatmaps.reshape(batch_size, 21, 2)

        accuracy_keypoint = ((true_keypoints - pred_keypoints)
                             ** 2).sum(axis=2) ** (1 / 2)
        accuracy_image = accuracy_keypoint.mean(axis=1)
        accuracy_all.extend(list(accuracy_image))

        # Calculate PCK@02
        avg_acc = batch_pck_calculation(
            pred_keypoints, true_keypoints, treshold=0.2, mask=None, normalize=None)
        pck_acc.append(avg_acc)

        # Calculate EPE mean and median, mind that it depends on what scale of input keypoints
        epe = batch_epe_calculation(pred_keypoints, true_keypoints)
        epe_lst.append(epe)

        # AUC calculation
        auc = batch_auc_calculation(
            pred_keypoints, true_keypoints, num_step=20, mask=None)
        auc_lst.append(auc)

    pck = sum(pck_acc) / len(pck_acc)
    epe_final = sum(epe_lst) / len(epe_lst)
    auc_final = sum(auc_lst) / len(auc_lst)

    print(f'PCK@2: {pck}, EPE: {epe_final}, AUC: {auc_final}')
    return accuracy_all, pck


# write main part of script
if __name__ == '__main__':

    model_pth = '/caa/Homes01/wmucha/repos/effhandegonet/saved_models/EffHandNet_FreiHAND_128x128.pth'

    config = {
        "data_dir": "/data/wmucha/datasets/freihand",
        "model_path": 'asdf',
        "batch_size": 64,
        "device": 0,
        'num_workers': 16,
    }

    dataloader = get_FreiHAND_dataloaders(config)

    model = EffHandNet(1280, 21)
    model.load_state_dict(
        torch.load(model_pth, map_location=torch.device(config["device"]))
    )
    model.eval()

    accuracy_all, pck = evaluate(model, dataloader['test'])
    accuracy_all, pck = evaluate(model, dataloader['test_final'])
