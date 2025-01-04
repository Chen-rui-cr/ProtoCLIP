import os

import PIL.Image as Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import scipy.io as sio


def map_label(label, classes):
    mapped_label = torch.LongTensor(len(label))
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label


def prepare_attri_label(attribute, classes):
    # print("attribute.shape", attribute.shape)
    classes_dim = classes.size(0)
    attri_dim = attribute.shape[1]
    output_attribute = torch.FloatTensor(classes_dim, attri_dim)
    for i in range(classes_dim):
        output_attribute[i] = attribute[classes[i]]
    return torch.transpose(output_attribute, 1, 0)


class CUB_dataset(Dataset):
    def __init__(
        self, datasets_path, image_path, label, image_attr, transforms, 
        train=True, train_type="nature", val_type="cartoon"
    ):
        super().__init__()
        self.datasets_path = datasets_path
        self.image_path = image_path
        self.label = label
        self.image_attr = image_attr

        self.transforms = transforms
        self.train = train
        self.train_type = train_type
        self.val_type = val_type

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        if self.train:
            if self.train_type == "nature":
                image = Image.open(os.path.join(self.datasets_path, str(self.image_path[idx])[56:-2]))
            elif self.train_type == "cartoon":
                image = Image.open(os.path.join(self.datasets_path, "cartoon_images",  str(self.image_path[idx])[63:-2]))
        else:
            if self.val_type == "nature":
                image = Image.open(os.path.join(self.datasets_path, str(self.image_path[idx])[56:-2]))
            elif self.val_type == "cartoon":
                image = Image.open(os.path.join(self.datasets_path, "cartoon_images", str(self.image_path[idx])[63:-2]))
        
        image = self.transforms(image)
        label = self.label[idx]

        return image, label


def CUB_dataloader(
        args
    ):
    matcontent = sio.loadmat(os.path.join(args.data_split_path, "res101.mat"))
    label = matcontent['labels'].astype(int).squeeze() - 1
    image_files = matcontent['image_files'].squeeze()

    matcontent = sio.loadmat(os.path.join(args.data_split_path, "att_splits.mat"))
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

    allclasses_name = matcontent['allclasses_names']
    attribute = torch.from_numpy(matcontent['att'].T).float()
    attri_name = matcontent['attri_name']

    train_label = torch.from_numpy(label[trainval_loc]).long()
    test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
    test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

    seenclasses = torch.from_numpy(np.unique(test_seen_label.numpy()))
    unseenclasses = torch.from_numpy(np.unique(test_unseen_label.numpy()))
    seenclasses_name = allclasses_name[seenclasses]
    unseenclasses_name = allclasses_name[unseenclasses]

    ntrain = train_label.size()[0]
    ntest_unseen = test_unseen_label.size()[0]
    ntest_seen = test_seen_label.size()[0]
    ntrain_class = seenclasses.size(0)
    ntest_class = unseenclasses.size(0)
    train_class = seenclasses.clone()
    allclasses = torch.arange(0, ntrain_class + ntest_class).long()

    train_mapped_label = map_label(train_label, seenclasses)
    test_mapped_label = map_label(test_unseen_label, unseenclasses)

    BICUBIC = transforms.InterpolationMode.BICUBIC
    resolution = args.resolution
    train_transforms = transforms.Compose([
        transforms.Resize(resolution, interpolation=BICUBIC),
        transforms.CenterCrop(resolution),
        lambda image: image.convert("RGB"),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.2, 2), contrast=(0.3, 1.5), saturation=(0.2, 2)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(resolution, interpolation=BICUBIC),
        transforms.CenterCrop(resolution),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # attribution
    attr = []
    for line in attri_name:
        discr = line.strip()
        a = []
        a.append(discr.split("::")[-1])
        a.append(discr.split("_")[1])
        if len(discr.split("_")) > 2:
            a.append(discr.split("_")[2].split(":")[0])
        # a.append(":")
        attr.append(" ".join(a))

    image_name = []
    image_attr = {}
    with open(os.path.join(args.datasets_path, "images.txt"), "r") as file:
        for line in file:
            _, name = line.strip().split(" ")
            image_name.append(name)

    with open(os.path.join(args.datasets_path, "attributes/image_attribute_idx.txt"), 'r') as file:
        for i, line in enumerate(file):
            int_list = [int(i)-1 for i in line.strip().split()]
            image_attr[image_name[i]] = int_list

    attribute_zsl = prepare_attri_label(attribute, unseenclasses).half().T.cuda()
    attribute_seen = prepare_attri_label(attribute, seenclasses).half().T.cuda()
    attribute_gzsl = torch.transpose(attribute, 1, 0).half().T.cuda()

    label_attr_sentence = []
    pre = "a photo of bird, with"
    for i in range(len(attribute_seen)):
        row_tensor = attribute_seen[i]
        non_zero_indices = torch.nonzero(row_tensor).squeeze()
        non_zero_values = row_tensor[non_zero_indices]
        values, indices = torch.topk(non_zero_values, k=min(20, len(non_zero_values)))
        top_indices = non_zero_indices[indices]
        sen = ""
        for ind in top_indices:
            sen += " "
            sen += attr[ind]
        label_attr_sentence.append(pre + sen)

    train_image_files = image_files[trainval_loc]
    test_seen_image_files = image_files[test_seen_loc]
    test_unseen_image_files = image_files[test_unseen_loc]

    train_ds = CUB_dataset(args.datasets_path, train_image_files, train_mapped_label, 
                           image_attr, train_transforms, train=True, 
                           train_type=args.train_type, val_type=args.val_type)
    test_zsl = CUB_dataset(args.datasets_path, test_unseen_image_files, test_mapped_label, 
                           image_attr, val_transforms, train=False, 
                           train_type=args.train_type, val_type=args.val_type)
    test_seen_ds = CUB_dataset(args.datasets_path, test_seen_image_files, test_seen_label, 
                               image_attr, val_transforms, train=False,
                               train_type=args.train_type, val_type=args.val_type)
    test_unseen_ds = CUB_dataset(args.datasets_path, test_unseen_image_files, test_unseen_label, 
                                 image_attr, val_transforms, train=False,
                                 train_type=args.train_type, val_type=args.val_type)

    return (train_ds, test_zsl, test_seen_ds, test_unseen_ds, attr, seenclasses_name, unseenclasses_name, attribute_zsl,
            attribute_seen, attribute_gzsl, label_attr_sentence, test_seen_label)


class SUN_dataset(Dataset):
    def __init__(
        self, datasets_path, image_path, label, image_attr, transforms, 
        train=True, train_type="nature", val_type="cartoon"
    ):
        super().__init__()
        self.datasets_path = datasets_path
        self.image_path = image_path
        self.label = label
        self.image_attr = image_attr

        self.transforms = transforms
        self.train = train
        self.train_type = train_type
        self.val_type = val_type

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        if self.train:
            if self.train_type == "nature":
                image = Image.open(os.path.join(self.datasets_path, str(self.image_path[idx])[39:-2]))
            elif self.train_type == "cartoon":
                image = Image.open(os.path.join(self.datasets_path, "cartoon_images", str(self.image_path[idx])[46:-2]))
        else:
            if self.val_type == "nature":
                image = Image.open(os.path.join(self.datasets_path, str(self.image_path[idx])[39:-2]))
            elif self.val_type == "cartoon":
                image = Image.open(os.path.join(self.datasets_path, "cartoon_images", str(self.image_path[idx])[46:-2]))
            elif self.val_type == "1900s":
                image = Image.open(os.path.join(self.datasets_path, "1900_images", str(self.image_path[idx])[46:-2]))
            elif self.val_type == "clay": 
                image = Image.open(os.path.join(self.datasets_path, "clay_images", str(self.image_path[idx])[46:-2]))
            elif self.val_type == "comic": 
                image = Image.open(os.path.join(self.datasets_path, "comic_images", str(self.image_path[idx])[46:-2]))

        image = self.transforms(image)
        label = self.label[idx]

        return image, label


def SUN_dataloader(args):
    matcontent = sio.loadmat(os.path.join(args.data_split_path, "res101.mat"))
    label = matcontent['labels'].astype(int).squeeze() - 1
    image_files = matcontent['image_files'].squeeze()

    matcontent = sio.loadmat(os.path.join(args.data_split_path, "att_splits.mat"))
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

    allclasses_name = matcontent['allclasses_names']
    attribute = torch.from_numpy(matcontent['att'].T).float()
    attri_name = matcontent['attri_name']

    train_label = torch.from_numpy(label[trainval_loc]).long()
    test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
    test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

    seenclasses = torch.from_numpy(np.unique(test_seen_label.numpy()))
    unseenclasses = torch.from_numpy(np.unique(test_unseen_label.numpy()))
    seenclasses_name = allclasses_name[seenclasses]
    unseenclasses_name = allclasses_name[unseenclasses]

    ntrain = train_label.size()[0]
    ntest_unseen = test_unseen_label.size()[0]
    ntest_seen = test_seen_label.size()[0]
    ntrain_class = seenclasses.size(0)
    ntest_class = unseenclasses.size(0)
    train_class = seenclasses.clone()
    allclasses = torch.arange(0, ntrain_class + ntest_class).long()

    train_mapped_label = map_label(train_label, seenclasses)
    test_unseen_mapped_label = map_label(test_unseen_label, unseenclasses)

    BICUBIC = transforms.InterpolationMode.BICUBIC
    resolution = args.resolution
    train_transforms = transforms.Compose([
        transforms.Resize(resolution, interpolation=BICUBIC),
        transforms.CenterCrop(resolution),
        lambda image: image.convert("RGB"),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.2, 2), contrast=(0.3, 1.5), saturation=(0.2, 2)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(resolution, interpolation=BICUBIC),
        transforms.CenterCrop(resolution),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # attribution
    attr = []
    for line in attri_name:
        discr = line.strip()
        attr.append(discr)

    attribute_zsl = prepare_attri_label(attribute, unseenclasses).half().T.cuda()
    attribute_seen = prepare_attri_label(attribute, seenclasses).half().T.cuda()
    attribute_gzsl = torch.transpose(attribute, 1, 0).half().T.cuda()

    label_attr_sentence = []
    pre = "a photo of scene, with"
    for i in range(len(attribute_seen)):
        row_tensor = attribute_seen[i]
        non_zero_indices = torch.nonzero(row_tensor).squeeze()
        non_zero_values = row_tensor[non_zero_indices]
        values, indices = torch.topk(non_zero_values, k=min(20, len(non_zero_values)))
        top_indices = non_zero_indices[indices]
        sen = ""
        for ind in top_indices:
            sen += " "
            sen += attr[ind]
        label_attr_sentence.append(pre + sen)

    train_image_files = image_files[trainval_loc]
    test_seen_image_files = image_files[test_seen_loc]
    test_unseen_image_files = image_files[test_unseen_loc]

    train_ds = SUN_dataset(args.datasets_path, train_image_files, train_mapped_label, 
                           attribute, train_transforms, train=True,
                           train_type=args.train_type, val_type=args.val_type)
    test_zsl = SUN_dataset(args.datasets_path, test_unseen_image_files, test_unseen_mapped_label, 
                           attribute, val_transforms, train=False,
                           train_type=args.train_type, val_type=args.val_type)
    test_seen_ds = SUN_dataset(args.datasets_path, test_seen_image_files, test_seen_label, 
                               attribute, val_transforms, train=False,
                               train_type=args.train_type, val_type=args.val_type)
    test_unseen_ds = SUN_dataset(args.datasets_path, test_unseen_image_files, test_unseen_label, 
                                 attribute, val_transforms, train=False,
                                 train_type=args.train_type, val_type=args.val_type)

    return (train_ds, test_zsl, test_seen_ds, test_unseen_ds, attr, seenclasses_name, unseenclasses_name, attribute_zsl,
            attribute_seen, attribute_gzsl, label_attr_sentence, test_seen_label)


class AWA2_dataset(Dataset):
    def __init__(
        self, datasets_path, image_path, label, image_attr, transforms, 
        train=True, train_type="nature", val_type="cartoon"
    ):
        super().__init__()
        self.datasets_path = datasets_path
        self.image_path = image_path
        self.label = label
        self.image_attr = image_attr

        self.transforms = transforms
        self.train = train
        self.train_type = train_type
        self.val_type = val_type

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        if self.train:
            if self.train_type == "nature":
                image = Image.open(os.path.join(self.datasets_path, str(self.image_path[idx])[46:-2]))
            elif self.train_type == "cartoon":
                image = Image.open(os.path.join(self.datasets_path, "cartoon_JPEGImages", str(self.image_path[idx])[57:-2]))
        else:
            if self.val_type == "nature":
                image = Image.open(os.path.join(self.datasets_path, str(self.image_path[idx])[46:-2]))
            elif self.val_type == "cartoon":
                image = Image.open(os.path.join(self.datasets_path, "cartoon_JPEGImages", str(self.image_path[idx])[57:-2]))

        image = self.transforms(image)
        label = self.label[idx]
     
        return image, label


def AWA2_dataloader(args=None):
    matcontent = sio.loadmat(os.path.join(args.data_split_path, "res101.mat"))
    label = matcontent['labels'].astype(int).squeeze() - 1
    image_files = matcontent['image_files'].squeeze()

    matcontent = sio.loadmat(os.path.join(args.data_split_path, "att_splits.mat")) 
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

    allclasses_name = matcontent['allclasses_names']
    attribute = torch.from_numpy(matcontent['att'].T).float()
    attri_name = matcontent['attri_name']

    train_label = torch.from_numpy(label[trainval_loc]).long()
    test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
    test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

    seenclasses = torch.from_numpy(np.unique(test_seen_label.numpy()))
    unseenclasses = torch.from_numpy(np.unique(test_unseen_label.numpy()))
    seenclasses_name = allclasses_name[seenclasses]
    unseenclasses_name = allclasses_name[unseenclasses]

    ntrain = train_label.size()[0]
    ntest_unseen = test_unseen_label.size()[0]
    ntest_seen = test_seen_label.size()[0]
    ntrain_class = seenclasses.size(0)
    ntest_class = unseenclasses.size(0)
    train_class = seenclasses.clone()
    allclasses = torch.arange(0, ntrain_class + ntest_class).long()

    train_mapped_label = map_label(train_label, seenclasses)
    test_unseen_mapped_label = map_label(test_unseen_label, unseenclasses)

    BICUBIC = transforms.InterpolationMode.BICUBIC
    resolution = args.resolution
    train_transforms = transforms.Compose([
        transforms.Resize(resolution, interpolation=BICUBIC),
        transforms.CenterCrop(resolution),
        lambda image: image.convert("RGB"),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.2, 2), contrast=(0.3, 1.5), saturation=(0.2, 2)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(resolution, interpolation=BICUBIC),
        transforms.CenterCrop(resolution),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # attribution
    attr = []
    for line in attri_name:
        discr = line.strip()
        attr.append(discr)

    attribute_zsl = prepare_attri_label(attribute, unseenclasses).half().T.cuda()
    attribute_seen = prepare_attri_label(attribute, seenclasses).half().T.cuda()
    attribute_gzsl = torch.transpose(attribute, 1, 0).half().T.cuda()

    label_attr_sentence = []
    pre = "a photo of animal, with "
    for i in range(len(attribute_seen)):
        row_tensor = attribute_seen[i]
        non_zero_indices = torch.nonzero(row_tensor).squeeze()
        non_zero_values = row_tensor[non_zero_indices]
        values, indices = torch.topk(non_zero_values, k=min(20, len(non_zero_values)))
        top_indices = non_zero_indices[indices]
        sen = ""

        for ind in top_indices:
            sen += " "
            sen += attr[ind]
        label_attr_sentence.append(pre + sen)

    train_image_files = image_files[trainval_loc]
    test_seen_image_files = image_files[test_seen_loc]
    test_unseen_image_files = image_files[test_unseen_loc]

    train_ds = AWA2_dataset(args.datasets_path, train_image_files, train_mapped_label, 
                            attribute, train_transforms, train=True,
                            train_type=args.train_type, val_type=args.val_type)
    test_zsl = AWA2_dataset(args.datasets_path, test_unseen_image_files, test_unseen_mapped_label, 
                            attribute, val_transforms, train=False,
                            train_type=args.train_type, val_type=args.val_type)
    test_seen_ds = AWA2_dataset(args.datasets_path, test_seen_image_files, test_seen_label, 
                                attribute, val_transforms, train=False,
                                train_type=args.train_type, val_type=args.val_type)
    test_unseen_ds = AWA2_dataset(args.datasets_path, test_unseen_image_files, test_unseen_label, 
                                  attribute, val_transforms, train=False,
                                  train_type=args.train_type, val_type=args.val_type)

    return (train_ds, test_zsl, test_seen_ds, test_unseen_ds, attr, seenclasses_name, unseenclasses_name, attribute_zsl,
            attribute_seen, attribute_gzsl, label_attr_sentence, test_seen_label)
