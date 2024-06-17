import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from vision_ensemble import AnomalyCLIP_VisionLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset, DatasetMedical
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform, encode_text_with_prompt_ensemble
from prompt import REAL_NAME

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}
CLASS_INDEX_INV = {3:'Brain', 2:'Liver', 1:'Retina_RESC', -1:'Retina_OCT2017', -2:'Chest', -3:'Histopathology'}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):

    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    train_data = DatasetMedical(root=args.train_data_path, batch_size=args.batch_size, img_size=args.image_size, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    ##########################################################################################
    vision_learner = AnomalyCLIP_VisionLearner(features=[6, 12, 18, 24])
    vision_learner.to(device)
    for name, param in vision_learner.named_parameters():
        param.requires_grad = True
    ##########################################################################################
    seg_optimizer = torch.optim.Adam(list(vision_learner.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(vision_learner.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    text_feature_list = [0]
    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        for i in [1, 2, 3, -3, -2, -1]:
            text_feature = encode_text_with_prompt_ensemble(model, REAL_NAME[CLASS_INDEX_INV[i]], device)
            text_feature_list.append(text_feature)

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    model.eval()
    vision_learner.train()
    for epoch in tqdm(range(args.epoch), position=0, leave=True):
        model.eval()
        loss_list = []
        image_loss_list = []

        for items in tqdm(train_dataloader, position=0, leave=True):
            image = items['img'].squeeze().to("cuda")
            label = items['anomaly']
            label = torch.cat(label, dim=0)
            cls_idx = items['cls_idx'].item()

            # Apply DPAM to the layer from 6 to 24
            # DPAM_layer represents the number of layer refined by DPAM from top to bottom
            # DPAM_layer = 1, no DPAM is used
            # DPAM_layer = 20 as default
            with torch.cuda.amp.autocast():
                _, [ori_det_patch_features, ori_seg_patch_features] = model.encode_image(image, args.features_list, DPAM_layer = 20)
                det_patch_features, seg_patch_features = vision_learner.encoder_vision(det_patch_features=ori_det_patch_features, seg_patch_features=ori_seg_patch_features)
                seg_patch_tokens = [p[:, 1:, :] for p in seg_patch_features]
                det_patch_tokens = [p[:, 1:, :] for p in det_patch_features]

                # Apply DPAM surgery
                det_loss = 0
                image_label = label.squeeze(0).to(device).float()
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] = det_patch_tokens[layer] / det_patch_tokens[layer].norm(dim=-1,
                                                                                                     keepdim=True)
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_feature_list[cls_idx])
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    det_loss += loss_bce(anomaly_score, image_label)
                image_loss_list.append(det_loss.item())

                if cls_idx > 0:
                    # pixel level
                    mask = items['img_mask'].squeeze().to(device)
                    mask[mask > 0.5] = 1
                    mask[mask <= 0.5] = 0

                    seg_loss = 0
                    for layer in range(len(seg_patch_tokens)):
                        seg_patch_tokens[layer] = seg_patch_tokens[layer] / seg_patch_tokens[layer].norm(dim=-1,
                                                                                                         keepdim=True)
                        # print(seg_patch_tokens[layer].shape, text_feature_list[seg_idx].shape) # torch.Size([289, 768]) torch.Size([768, 2])
                        anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_feature_list[cls_idx])
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.image_size, mode='bilinear', align_corners=True)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)

                    loss = seg_loss + det_loss  # = focal(seg_out, mask) + bce(det_out, y)
                    loss.requires_grad_(True)
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    loss.backward()
                    seg_optimizer.step()
                    det_optimizer.step()
                    loss_list.append(loss.item())
                else:
                    loss = det_loss
                    loss.requires_grad_(True)
                    det_optimizer.zero_grad()
                    loss.backward()
                    det_optimizer.step()

        train_data.shuffle_dataset()
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"vision_learner": vision_learner.state_dict()}, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/medical", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')


    parser.add_argument("--dataset", type=str, default='medical', help="train dataset name")

    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--image_size", type=int, default=240, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
