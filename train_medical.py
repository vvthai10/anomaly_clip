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
from utils import get_transform
from scipy.ndimage import gaussian_filter
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


    #########################################################################################
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.to(device)
    for name, param in prompt_learner.named_parameters():
        param.requires_grad = True

    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)
    ##########################################################################################
    optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))


    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()


    model.eval()
    prompt_learner.train()
    for epoch in tqdm(range(args.epoch), position=0, leave=True):
        model.eval()
        prompt_learner.train()
        loss_list = []
        image_loss_list = []

        for items in tqdm(train_dataloader, position=0, leave=True):
            image = items['img'].squeeze().to("cuda")
            label = items['anomaly']
            label = torch.cat(label, dim=0)
            cls_idx = items['cls_idx'].item()

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            # Apply DPAM to the layer from 6 to 24
            # DPAM_layer represents the number of layer refined by DPAM from top to bottom
            # DPAM_layer = 1, no DPAM is used
            # DPAM_layer = 20 as default
            image_features, seg_patch_features = model.encode_image(image, args.features_list, DPAM_layer = 20)

            ####################################
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)

            image_loss = 0
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            anomaly_map = image_features @ text_features.permute(0, 2, 1)
            anomaly_map = (anomaly_map / 0.07)
            anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
            anomaly_score = torch.mean(anomaly_map, dim=-1)
            image_loss += loss_bce(anomaly_score, label.to(device).float())

            if cls_idx > 0:
                similarity_map_list = []
                for idx, patch_feature in enumerate(seg_patch_features):
                    if idx >= args.feature_map_layer[0]:
                        patch_feature = patch_feature / patch_feature.norm(dim = -1, keepdim = True)
                        similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                        similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                        similarity_map_list.append(similarity_map)

                loss = 0
                for i in range(len(similarity_map_list)):
                    loss += loss_focal(similarity_map_list[i], gt)
                    loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                    loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1-gt)

                total_loss = loss+image_loss
                total_loss.requires_grad_(True)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            else:
                total_loss = image_loss
                total_loss.requires_grad_(True)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        train_data.shuffle_dataset()
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"prompt_learner": prompt_learner.state_dict()}, ckp_path)

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
