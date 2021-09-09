import sys
import os
os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt" # A workaround in case this happens: https://github.com/mapbox/rasterio/issues/1289
import time
import datetime
import argparse
import copy

import numpy as np
import pandas as pd

from dataloaders.StreamingDatasets import StreamingGeospatialDataset

import torch
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import models
import utils

from azureml.core import Run
run = Run.get_context()

NUM_WORKERS = 4
CHIP_SIZE = 256

parser = argparse.ArgumentParser(description='DFC2021 baseline training script')
parser.add_argument('--input_fn', type=str, required=True, help='The path to a CSV file containing three columns -- "image_fn", "label_fn", and "group" -- that point to tiles of imagery and labels as well as which "group" each tile is in.')
parser.add_argument('--input_fn_val', type=str, required=True,  help='The path to a CSV file containing three columns -- "image_fn", "label_fn", and "group" -- that point to tiles of imagery and labels as well as which "group" each tile is in.')
parser.add_argument('--output_dir', type=str, required=True,  help='The path to a directory to store model checkpoints.')
parser.add_argument('--overwrite', action="store_true",  help='Flag for overwriting `output_dir` if that directory already exists.')
parser.add_argument('--save_most_recent', action="store_true",  help='Flag for saving the most recent version of the model during training.')
parser.add_argument('--model', default='fcn',
    choices=(
        'unet',
        'fcn',
        'unet2'
    ),
    help='Model to use'
)

## Training arguments
parser.add_argument('--gpu', type=int, default=0, help='The ID of the GPU to use')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use for training')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
parser.add_argument('--seed', type=int, default=0, help='Random seed to pass to numpy and torch')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes in dataset')
parser.add_argument('--num_chips', type=int, default=100, help='number of chips to randomly sample from data')
parser.add_argument('--label_transform', default='uvm', help='str either naip, epa or cic, naip_5cls, uvm to indicate how to transform labels')
args = parser.parse_args()

def image_transforms(img, group):
    img = img/255.0
    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = torch.from_numpy(img)
    return img

def label_transforms_naip(labels, group):
    labels = np.array(labels).astype(np.int64)
    labels = np.where(labels == 14, 0, labels) #to no data
    labels = np.where(labels == 15, 0, labels)  # to no data
    labels = np.where(labels == 13, 0, labels) #to no data
    labels = np.where(labels == 10, 3, labels) # to tree canopy
    labels = np.where(labels == 11, 3, labels) # to tree canopy
    labels = np.where(labels == 12, 3, labels) # to tree canopy
    #labels = utils.get_lc_class_to_idx_map()
    labels = torch.from_numpy(labels)
    return labels

def label_transforms_epa(labels, group):
    #labels = utils.EPA_CLASS_TO_IDX_MAP[labels]
    labels = np.array(labels).astype(np.int64)
    labels_new = np.copy(labels)
    for k, v in utils.epa_label_dict.items():
        labels_new[labels==k] = v
    labels_new = torch.from_numpy(labels_new)
    return labels_new

def label_transform_cic(labels, group):
    labels = np.array(labels).astype(np.int64)
    labels_new = np.copy(labels)
    for k, v in utils.cic_label_dict.items():
        labels_new[labels==k] = v
    labels_new = torch.from_numpy(labels_new)
    return labels_new

def label_transform_naip5cls(labels, group):
    labels = np.array(labels).astype(np.int64)
    labels_new = np.copy(labels)
    for k, v in utils.naip_5cls.items():
        labels_new[labels==k] = v
    labels_new = torch.from_numpy(labels_new)
    return labels_new

def label_transform_4cls(labels, group):
    labels = np.array(labels).astype(np.int64)
    labels_new = np.copy(labels)
    for k, v in utils.naip_4cls.items():
        labels_new[labels==k] = v
    labels_new = torch.from_numpy(labels_new)
    return labels_new

def labels_transform_uvm(labels, group):
    labels = np.array(labels).astype(np.int64)
    labels_new = np.copy(labels)
    for k, v in utils.uvm_7cls.items():
        labels_new[labels==k] = v
    labels_new = torch.from_numpy(labels_new)
    return labels_new

def nodata_check(img, labels):
    # print(np.unique(labels, return_counts=True)[1])
    # if 0 in np.unique(labels, return_counts=True)[0]:
    #     black_prop = (np.unique(labels, return_counts = True)[1][0]) / (256 * 256)
    #     if black_prop >= 0.1:
    #         print('skipping chip')
    #         return True
    # else:
    #     return False
    return np.any(labels == 0)
    #return np.any(labels == 0) or np.any(labels == 13) or np.any(labels == 14) or np.any(labels == 15)



def main():
    print("Starting DFC2021 baseline training script at %s" % (str(datetime.datetime.now())))


    #-------------------
    # Setup
    #-------------------
    assert os.path.exists(args.input_fn)

    if os.path.isfile(args.output_dir):
        print("A file was passed as `--output_dir`, please pass a directory!")
        return

    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)):
        if args.overwrite:
            print("WARNING! The output directory, %s, already exists, we might overwrite data in it!" % (args.output_dir))
        else:
            print("The output directory, %s, already exists and isn't empty. We don't want to overwrite and existing results, exiting..." % (args.output_dir))
            return
    else:
        print("The output directory doesn't exist or is empty.")
        os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % args.gpu)
    else:
        print ("WARNING! Torch is reporting that CUDA isn't available, using cpu")
        device = 'cpu'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    #-------------------
    # Load input data
    #-------------------
    input_dataframe = pd.read_csv(args.input_fn)
    image_fns = input_dataframe["image_fn"].values
    label_fns = input_dataframe["label_fn"].values
    groups = input_dataframe["group"].values

    input_dataframe_val = pd.read_csv(args.input_fn_val)
    image_fns_val = input_dataframe_val["image_fn"].values
    label_fns_val = input_dataframe_val["label_fn"].values
    groups_val = input_dataframe_val["group"].values

    if args.label_transform == "naip":
        label_transform = label_transforms_naip
    elif args.label_transform == "epa":
        label_transform = label_transforms_epa
    elif args.label_transform == 'cic':
        label_transform = label_transform_cic
    elif args.label_transform == 'naip_5cls':
        label_transform = label_transform_naip5cls
    elif args.label_transform == 'naip_4cls':
        label_transform = label_transform_4cls
    elif args.label_transform == 'uvm':
        label_transform = labels_transform_uvm
    else:
        raise ValueError("Invalid label transform")



    dataset = StreamingGeospatialDataset(
        imagery_fns=image_fns, label_fns=label_fns, groups=groups, chip_size=CHIP_SIZE, num_chips_per_tile=args.num_chips, windowed_sampling=False, verbose=True,
        image_transform=image_transforms, label_transform=label_transform, nodata_check=nodata_check
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    dataset_val = StreamingGeospatialDataset(
        imagery_fns=image_fns_val, label_fns=label_fns_val, groups=groups, chip_size=CHIP_SIZE, num_chips_per_tile=args.num_chips, windowed_sampling=False, verbose=True,
        image_transform=image_transforms, label_transform=label_transform, nodata_check=nodata_check
    )

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    num_training_batches_per_epoch = int(len(image_fns) * args.num_chips / args.batch_size)
    print("We will be training with %d batches per epoch" % (num_training_batches_per_epoch))

    num_val_batches_per_epoch = int(len(image_fns_val) * args.num_chips / args.batch_size)
    print("We will be validating with %d batches per epoch" % (num_val_batches_per_epoch))

    #-------------------
    # Setup training
    #-------------------
    if args.model == "unet":
        model = models.get_unet(classes = args.num_classes)
    elif args.model == "unet2":
        model = models.get_unet2(n_classes=args.num_classes)
    elif args.model == "fcn":
        model = models.get_fcn(num_output_classes = args.num_classes)
    else:
        raise ValueError("Invalid model")

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    print("Model has %d parameters" % (utils.count_parameters(model)))


    #-------------------
    # Model training
    #-------------------
    training_task_losses = []
    num_times_lr_dropped = 0
    model_checkpoints = []
    val_task_losses = []
    temp_model_fn = os.path.join(args.output_dir, "most_recent_model.pt")

    for epoch in range(args.num_epochs):
        print('on epoch number: ', epoch)
        lr = utils.get_lr(optimizer)

        training_losses = utils.fit(
            model,
            device,
            dataloader,
            num_training_batches_per_epoch,
            optimizer,
            criterion,
            epoch,
        )
        scheduler.step(training_losses[0])

        model_checkpoints.append(copy.deepcopy(model.state_dict()))
        if args.save_most_recent:
            torch.save(model.state_dict(), temp_model_fn)

        if utils.get_lr(optimizer) < lr:
            num_times_lr_dropped += 1
            print("")
            print("Learning rate dropped")
            print("")
        training_task_losses.append(training_losses[0])
        run.log('loss', training_losses[0])
        if num_times_lr_dropped == 4:
            break

        # Run Validation
        validation_losses = utils.evaluate(model, device, dataloader_val, num_val_batches_per_epoch, criterion, epoch,)
        val_task_losses.append(validation_losses[0])
        run.log('loss', validation_losses[0])

        num_classes = args.num_classes #to-do fix
        per_class_f1, global_f1 = utils.score_batch(model, device, dataloader_val, num_val_batches_per_epoch, num_classes)
        run.log('per_class_f1_val', per_class_f1)
        run.log('global_f1_val', global_f1)



    #-------------------
    # Save everything
    #-------------------
    save_obj = {
        'args': args,
        'training_task_losses': training_task_losses,
        "checkpoints": model_checkpoints
    }

    save_obj_fn = "results.pt"
    out_path = os.path.join(args.output_dir, save_obj_fn)
    run.log('out_path', out_path)
    with open(os.path.join(args.output_dir, save_obj_fn), 'wb') as f:
        torch.save(save_obj, f)

if __name__ == "__main__":
    main()
