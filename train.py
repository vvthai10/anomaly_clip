import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from vision_ensemble import AnomalyCLIP_VisionLearner
from loss import FocalLoss, BinaryDiceLoss
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform


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

    # Setup Anomaly model
    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }

    model, _ = AnomalyCLIP_lib.load(
        "ViT-L/14@336px",
        device=device,
        design_details=AnomalyCLIP_parameters,
        training=True,
    )

    model.visual.DAPM_replace(DPAM_layer=20)

    # Load dataset
    train_data = Dataset(
        root=args.train_data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )

    # Set up Prompt Learner
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.to(device)
    for _, param in prompt_learner.named_parameters():
        param.requires_grad = True

    # Set up Vision Learner
    vision_learner = AnomalyCLIP_VisionLearner(features=args.features_list)
    vision_learner.to(device)
    for name, param in vision_learner.named_parameters():
        param.requires_grad = True

    # Setup Optimizer
    text_optimizer = torch.optim.Adam(
        list(prompt_learner.parameters()), lr=0.001, betas=(0.5, 0.999)
    )

    vision_optimizer = torch.optim.Adam(
        list(vision_learner.parameters()), lr=0.0001, betas=(0.5, 0.999)
    )

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    # Set to evaluated model // frozen
    model.eval()
    model.to(device)

    # Train prompt learner & vision learner
    prompt_learner.train()
    vision_learner.train()

    # Per epoch
    for epoch in tqdm(range(args.epoch), position=0, leave=True):
        loss_list = []
        image_loss_list = []

        # Per items
        for items in tqdm(train_dataloader, position=0, leave=True):
            image = items["img"].to(device)  # b, 3, h, w => 2, 3, 240, 240
            label = items["anomaly"]

            # Mask
            mask = items["img_mask"].squeeze().to(device)
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0

            # Get prompts token
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(
                cls_id=None
            )

            # Apply DPAM to the layer from 6 to 24
            # DPAM_layer represents the number of layer refined by DPAM from top to bottom
            # DPAM_layer = 1, no DPAM is used
            # DPAM_layer = 20 as default
            with torch.no_grad():
                # Vision features
                ori_image_features, ori_patch_features = model.encode_image(
                    image, args.features_list, DPAM_layer=20
                )
                image_features = ori_image_features / ori_image_features.norm(
                    dim=-1, keepdim=True
                )

            # Get segment features (4 x (2, 290, 768))
            det_patch_features, seg_patch_features = vision_learner.encoder_vision(
                ori_image_features, ori_patch_features
            )

            # print(len(det_patch_features), det_patch_features.shape)

            # Text features
            text_features = model.encode_text_learn(
                prompts, tokenized_prompts, compound_prompts_text
            )
            text_features = torch.stack(
                torch.chunk(text_features, dim=0, chunks=2), dim=1
            )

            # text feature: 1, 2, 768
            # image feature: 2, 768
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # text feature: 1, 768, 2
            # image feature: 2, 1, 768
            # text_probs: 2, 1, 2
            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = (text_probs[:, 0, ...] / 0.07).softmax(-1)

            # Classification loss
            glob_loss = loss_bce(torch.mean(text_probs, dim=(0)), label.to(device).float())

            loca_loss = []
            for idx, patch_feature in enumerate(det_patch_features):
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                similarity, _ = AnomalyCLIP_lib.compute_similarity(
                    patch_feature, text_features[0]
                )
                similarity = similarity.permute(0, 2, 1) # B, I, T -> B T I
                det_score = torch.mean(similarity, dim=(-1, 0))
                loca_loss.append(
                    loss_bce(
                        det_score,
                        label.to(device).float(),
                    )
                )
            loca_loss = torch.mean(torch.stack(loca_loss))
            image_loss = glob_loss + loca_loss
            # print("Image loss: ", image_loss)

            similarity_map_list = []
            for idx, patch_feature in enumerate(seg_patch_features):
                if idx >= args.feature_map_layer[0]:
                    # Normalize patch_feature
                    patch_feature = patch_feature / patch_feature.norm(
                        dim=-1, keepdim=True
                    )

                    # Compute similarity
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(
                        patch_feature, text_features[0]
                    )
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(
                        similarity[:, 1:, :], args.image_size
                    ).permute(0, 3, 1, 2)

                    similarity_map_list.append(similarity_map)

            loss = 0
            for i in range(len(similarity_map_list)):
                loss += loss_focal(similarity_map_list[i], mask)
                loss += loss_dice(similarity_map_list[i][:, 1, :, :], mask)
                loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1 - mask)
            # print("Loss: ", loss)
            # Caculate total loss

            total_loss = loss + image_loss
            total_loss.requires_grad_(True)

            # Resets the gradients of all optimized
            text_optimizer.zero_grad()
            vision_optimizer.zero_grad()

            # Backward
            total_loss.backward()

            # Performs a single optimization step
            text_optimizer.step()
            vision_optimizer.step()

            loss_list.append(loss.item())
            image_loss_list.append(image_loss.item())

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info(
                "epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}".format(
                    epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)
                )
            )

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, "epoch_" + str(epoch + 1) + ".pth")
            torch.save(
                {
                    "prompt_learner": prompt_learner.state_dict(),
                    "vision_learner": vision_learner.state_dict(),
                },
                ckp_path,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="./data/medical",
        help="train dataset path",
    )
    parser.add_argument(
        "--save_path", type=str, default="./checkpoint", help="path to save results"
    )

    parser.add_argument(
        "--dataset", type=str, default="medical", help="train dataset name"
    )

    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument(
        "--feature_map_layer",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="zero shot",
    )
    parser.add_argument(
        "--features_list",
        type=int,
        nargs="+",
        default=[6, 12, 18, 24],
        help="features used",
    )

    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--image_size", type=int, default=240, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
