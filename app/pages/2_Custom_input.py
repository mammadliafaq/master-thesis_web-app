"""Streamlit web app"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import transformers
import yaml
from PIL import Image
from sklearn.preprocessing import LabelEncoder

from app.dataset import ShopeeImageDataset, ShopeeTextDataset
from app.models.image_model import ShopeeImageModel
from app.models.text_model import ShopeeTextModel
from app.transforms import get_valid_transforms
from app.utils.eval_utils import (generate_image_embedding,
                                  generate_image_features,
                                  generate_text_embedding,
                                  generate_text_features,
                                  get_custom_concatenated_predictions,
                                  get_custom_image_predictions,
                                  get_custom_text_predictions)
from app.utils.utils import convert_dict_to_tuple, set_seed


@st.cache_resource
def load_tokenizer(config):
    print("Setting up the tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.text_model.model_name)
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


def main(args: argparse.Namespace):
    st.title(
        "Type in the description and upload the image of the item that you would like to find"
    )

    input_image = st.sidebar.file_uploader("Upload an image of the desired item")
    input_text = st.sidebar.text_input("Type in the description of the desired item")

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

    # Process input image
    if input_image is not None:
        "Input query image"
        file_bytes = np.asarray(bytearray(input_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption="Desired item")

        augmented = get_valid_transforms(config)(image=image)
        tensor_image = augmented["image"]
        tensor_image = tensor_image.unsqueeze(0)

        image_embedding = generate_image_embedding(image_model, tensor_image, device)

        # Only image
        if not input_text:
            image_predictions = get_custom_image_predictions(
                image_embedding, image_features
            )
            "Best predictions for input image: "
            print("Image predictions: ", image_predictions)
            if image_predictions[0].any():
                for idx, result_idx in enumerate(image_predictions[0]):
                    path_to_image = valid.iloc[result_idx]["filepath"]
                    image_to_show = Image.open(path_to_image)
                    st.image(
                        image_to_show,
                        caption=valid.iloc[result_idx]["title"],
                    )
        else:
            print(input_text)
            input_text_tokenized = tokenizer(
                input_text,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            input_ids = input_text_tokenized["input_ids"][0]
            attention_mask = input_text_tokenized["attention_mask"][0]

            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

            text_embedding = generate_text_embedding(
                text_model, input_ids, attention_mask, device
            )
            concatenated_embedding = np.concatenate(
                [image_embedding, text_embedding], axis=1
            )
            norm = np.linalg.norm(concatenated_embedding, axis=1)
            concatenated_embedding_normalized = concatenated_embedding / norm
            concatenated_predictions = get_custom_concatenated_predictions(
                concatenated_embedding_normalized, concatenated_features
            )
            "Best multimodal predictions: "
            print("Multimodal predictions: ", concatenated_predictions)
            if concatenated_predictions[0].any():
                for idx, result_idx in enumerate(concatenated_predictions[0]):
                    path_to_image = valid.iloc[result_idx]["filepath"]
                    image_to_show = Image.open(path_to_image)
                    st.image(
                        image_to_show,
                        caption=valid.iloc[result_idx]["title"],
                    )

    # Process input text
    if input_text:
        print(input_text)
        f"Input text: {input_text}"
        input_text_tokenized = tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        input_ids = input_text_tokenized["input_ids"][0]
        attention_mask = input_text_tokenized["attention_mask"][0]

        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

        text_embedding = generate_text_embedding(
            text_model, input_ids, attention_mask, device
        )

        if input_image is None:
            text_predictions = get_custom_text_predictions(
                text_embedding, text_features
            )
            "Best predictions for input text: "
            if text_predictions[0].any():
                for idx, result_idx in enumerate(text_predictions[0]):
                    path_to_image = valid.iloc[result_idx]["filepath"]
                    image_to_show = Image.open(path_to_image)
                    st.image(
                        image_to_show,
                        caption=valid.iloc[result_idx]["title"],
                    )


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="app/config/eval.yml", help="Path to config file."
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
