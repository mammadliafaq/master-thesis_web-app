"""Streamlit web app"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
from dataset import ShopeeImageDataset, ShopeeTextDataset
from models.image_model import ShopeeImageModel
from models.text_model import ShopeeTextModel
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from transforms import get_valid_transforms
from utils.eval_utils import (generate_image_features, generate_text_features,
                              get_concatenated_predictions, plot_threshold)
from utils.utils import convert_dict_to_tuple, set_seed


@st.cache_resource
def load_tokenizer(config):
    print("Setting up the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.text_model.model_name)
    return tokenizer


@st.cache_resource
def load_image_model(config, device):
    # Setting up the image model and checkpoint
    print("Creating model and loading checkpoint")
    # Define image model
    image_model_params = {
        "n_classes": config.dataset.num_of_classes,
        "model_name": config.image_model.model_name,
        "pretrained": config.image_model.pretrained,
        "use_fc": config.image_model.use_fc,
        "fc_dim": config.image_model.fc_dim,
        "dropout": config.image_model.dropout,
        "loss_module": config.image_model.loss_module,
    }
    image_model = ShopeeImageModel(**image_model_params, device=device)
    image_checkpoint = torch.load(
        config.image_model.path_to_weights, map_location="cuda"
    )
    image_model.load_state_dict(image_checkpoint)
    print("Image model weights have been loaded successfully.")
    image_model.to(device)
    return image_model


@st.cache_resource
def load_text_model(config, device):
    text_model_params = {
        "n_classes": config.dataset.num_of_classes,
        "model_name": config.text_model.model_name,
        "use_fc": config.text_model.use_fc,
        "fc_dim": config.text_model.fc_dim,
        "dropout": config.text_model.dropout,
        "loss_module": config.text_model.loss_module,
    }
    text_model = ShopeeTextModel(**text_model_params, device=device)

    text_checkpoint = torch.load(config.text_model.path_to_weights, map_location="cuda")

    text_model.load_state_dict(text_checkpoint)
    print("Text model weights have been loaded successfully.")
    text_model.to(device)
    return text_model


@st.cache_data
def load_valid_data(config):
    data = pd.read_csv(config.dataset.path_to_folds)
    data["filepath"] = data["image"].apply(
        lambda x: os.path.join(config.dataset.root, "train_images", x)
    )

    encoder = LabelEncoder()
    data["label_group"] = encoder.fit_transform(data["label_group"])

    valid = data[data["fold"] == 0].reset_index(drop=True)

    return valid


@st.cache_data
def load_image_dataloader(config, valid):
    image_valid_dataset = ShopeeImageDataset(
        csv=valid,
        transforms=get_valid_transforms(config),
    )

    image_valid_loader = torch.utils.data.DataLoader(
        image_valid_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=config.dataset.num_workers,
    )

    return image_valid_loader


@st.cache_data
def load_text_dataloader(config, valid, _tokenizer):
    text_valid_dataset = ShopeeTextDataset(
        csv=valid,
        tokenizer=_tokenizer,
    )

    text_valid_loader = torch.utils.data.DataLoader(
        text_valid_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=config.dataset.num_workers,
    )

    return text_valid_loader


@st.cache_data
def generate_image_embeddings(config, _image_model, _image_valid_loader, device):
    print("Generating image embeddings for the validation set to evaluate f1 score...")
    image_features = generate_image_features(
        config, _image_model, _image_valid_loader, device
    )
    return image_features


@st.cache_data
def generate_text_embeddings(_text_model, _text_valid_loader, device):
    print("Generating text embeddings for the validation set to evaluate f1 score...")
    text_features = generate_text_features(_text_model, _text_valid_loader, device)
    return text_features


@st.cache_data
def concatenate_features(image_features, text_features):
    concatenated_features = np.concatenate([image_features, text_features], axis=1)
    norm = np.linalg.norm(concatenated_features, axis=1).reshape(
        concatenated_features.shape[0], 1
    )
    concatenated_features_normalized = concatenated_features / norm
    return concatenated_features_normalized


@st.cache_data
def get_matches(valid, concatenated_features, config):
    f1_score, output_df = get_concatenated_predictions(
        valid, concatenated_features, threshold=config.threshold
    )
    return output_df


def main(args: argparse.Namespace):
    st.title("Welcome to the Afa shop!")
    "Select the item from the Shopee set or upload your own data"
    with open(args.cfg) as f:
        data = yaml.safe_load(f)

    config = convert_dict_to_tuple(data)

    # Set seed
    set_seed(config.seed)

    # Defining Device
    device_id = config.gpu_id
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print("Selected device: ", device)

    valid = load_valid_data(config)

    print("Validation shape: ", valid.shape)

    tokenizer = load_tokenizer(config)

    valid_image_loader = load_image_dataloader(config, valid)
    valid_text_loader = load_text_dataloader(config, valid, tokenizer)

    image_model = load_image_model(config, device)
    image_model.to(device)

    text_model = load_text_model(config, device)
    text_model.to(device)

    image_features = generate_image_embeddings(
        config, image_model, valid_image_loader, device
    )
    print("Image features: ", image_features.shape)
    text_features = generate_text_embeddings(text_model, valid_text_loader, device)
    print("Text features: ", text_features.shape)
    concatenated_features = concatenate_features(image_features, text_features)

    print("Concatenated features shape: ", concatenated_features.shape)

    output_df = get_matches(valid, concatenated_features, config)

    # output_df

    # output_df.to_csv('output_df.csv')

    # Test
    # width = st.sidebar.slider('Image width?', 1, 1000, 100)

    # Read input image
    option = st.sidebar.selectbox(
        "Select the index of the item: ", np.arange(valid.shape[0])
    )
    print(option)
    "Input image"
    path_to_input = output_df.iloc[int(option)]["filepath"]
    input_image = Image.open(path_to_input)
    st.image(input_image, caption=output_df.iloc[int(option)]["title"])

    # Show results
    results = output_df.iloc[int(option)]["pred_concat"]
    "Best matches: "
    for idx, result in enumerate(results):
        path_to_image = output_df.loc[
            output_df["posting_id"] == result, "filepath"
        ].iloc[0]
        image_to_show = Image.open(path_to_image)
        st.image(
            image_to_show,
            caption=output_df.loc[output_df["posting_id"] == result, "title"].iloc[0],
        )


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="app/config/eval.yml", help="Path to config file."
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
