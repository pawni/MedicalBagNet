import copy
import os
import shutil
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from data.ixi import IxiDataset
from data.camcan import CamCANDataset

from bagnets import bagnet9, bagnet17, bagnet33, bagnet177


def load_checkpoint(state, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        state.update(checkpoint)
        print('Loaded checkpoint')

        return True
    return False


def evaluate(model, criterion, loader, classification=True):
    model.eval()

    eval_stats = {}

    # Test the model
    with torch.no_grad():
        total_loss = 0
        n = 0

        if classification:
            correct = []
            all_probs = []
        else:
            aes = []

        for images, labels, _ in tqdm(loader):
            if torch.cuda.is_available():
                if torch.cuda.device_count() == 1:
                    images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)

            if classification:
                loss = criterion(outputs, labels)
                predicted = torch.argmax(outputs.data, 1)
                correct += list((predicted.detach() == labels).cpu())

                all_probs += list(F.softmax(outputs, 1).data.cpu().numpy())
            else:
                loss = criterion(outputs.squeeze(), labels)
                cur_ae = torch.abs(outputs.detach().squeeze()
                                   - labels).cpu().numpy()
                aes += list(cur_ae)

            total_loss += loss.mean().cpu().item()
            n += 1

        eval_stats['loss'] = total_loss / n
        if classification:
            eval_stats['accuracy'] = np.mean(correct)
            eval_stats['probs'] = np.array(all_probs)
        else:
            eval_stats['mae'] = np.mean(aes) * IxiDataset.age_std
            eval_stats['aes'] = np.array(aes).reshape(
                (-1,)) * IxiDataset.age_std

        return eval_stats

def localise(model, criterion, loader, output_path, classification=True):
    import SimpleITK as sitk
    model.eval()

    eval_stats = {}

    # Test the model
    with torch.no_grad():
        total_loss = 0
        n = 0

        if classification:
            correct = []
        else:
            aes = []

        for images, labels, id in tqdm(loader):
            save_path = os.path.join(output_path, '{}.nii.gz'.format(id[0]))
            if torch.cuda.is_available():
                if torch.cuda.device_count() == 1:
                    images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            global_output = outputs.mean(1).mean(1).mean(1)

            if classification:
                loss = criterion(global_output, labels)
                predicted = torch.argmax(global_output.data, 1)
                correct += list((predicted.detach() == labels).cpu())

                probs = F.softmax(outputs, 4).data[...,0].cpu().numpy().squeeze()
                sitk_img = sitk.GetImageFromArray(probs)
                sitk.WriteImage(sitk_img, save_path)
            else:
                global_output = global_output[:, 0]
                loss = criterion(global_output, labels)
                cur_ae = torch.abs(global_output.detach().squeeze()
                                   - labels).cpu().numpy()
                aes += list(cur_ae)

                age_pred = outputs.cpu().numpy().squeeze()
                sitk_img = sitk.GetImageFromArray(age_pred)
                sitk.WriteImage(sitk_img, save_path)

            total_loss += loss.mean().cpu().item()
            n += 1

        eval_stats['loss'] = total_loss / n
        if classification:
            eval_stats['accuracy'] = np.mean(correct)
        else:
            eval_stats['mae'] = np.mean(aes)
            eval_stats['aes'] = np.array(aes).reshape((-1,))

        return eval_stats


def get_datasets(data_type, data_path, attribute='sex', scale='2mm'):
    if data_type == 'ixi':
        train_dataset = IxiDataset(
            data_path, os.path.join(data_path, 'train.csv'), attribute=attribute,
            augment=True, scale=scale)

        val_dataset = IxiDataset(
            data_path, os.path.join(data_path, 'val.csv'), attribute=attribute,
            scale=scale)

        test_dataset = IxiDataset(
            data_path, os.path.join(data_path, 'test.csv'), attribute=attribute,
            scale=scale)
    elif data_type == 'camcan':
        train_dataset = CamCANDataset(
            os.path.join(data_path, 'train.csv'),
            attribute=attribute, center_crop=True)

        val_dataset = CamCANDataset(
            os.path.join(data_path, 'val.csv'), attribute=attribute, center_crop=True)

        test_dataset = CamCANDataset(
            os.path.join(data_path, 'test.csv'), attribute=attribute, center_crop=True)
    else:
        raise Exception('unknown data type')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=2,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=2, shuffle=False)

    return train_loader, val_loader, test_loader

def deploy(args):

    print('Starting run with:\n{}'.format(args))

    train_loader, val_loader, test_loader = get_datasets(
        args.data_type, args.data_path, args.attribute,
        args.scale)

    if args.rf == 9:
        model = bagnet9(num_classes=args.num_classes,
                        scale_filters=args.scale_factor,
                        avg_pool=not args.localised)
    elif args.rf == 17:
        model = bagnet17(num_classes=args.num_classes,
                         scale_filters=args.scale_factor,
                         avg_pool=not args.localised)
    elif args.rf == 33:
        model = bagnet33(num_classes=args.num_classes,
                         scale_filters=args.scale_factor,
                         avg_pool=not args.localised)
    elif args.rf == 177:
        model = bagnet177(num_classes=args.num_classes,
                          scale_filters=args.scale_factor,
                          avg_pool=not args.localised)
    print(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raw_model = model
    if torch.cuda.device_count() > 1:
        print('using multiple gpus')
        model = torch.nn.DataParallel(model)
    model.to(device)

    if args.attribute == 'sex':
        criterion = nn.CrossEntropyLoss()
        classification = True
    elif args.attribute == 'age':
        criterion = nn.MSELoss()
        classification = False
    else:
        raise Exception('attribute not known')
    print(criterion)


    state = torch.load(args.model_path)
    raw_model.load_state_dict(state['state_dict'])

    if args.localised:
        val_stats = localise(model, criterion, val_loader, args.save_path,
                             classification)

        train_stats = localise(model, criterion, train_loader, args.save_path,
                               classification)

        test_stats = localise(model, criterion, test_loader, args.save_path,
                              classification)

        print('done...')
        if classification:
            print('Train: Loss {} Acc {}'.format(
                train_stats['loss'], train_stats['accuracy']))
            print('Val: Loss {} Acc {}'.format(
                val_stats['loss'], val_stats['accuracy']))
            print('Test: Loss {} Acc {}'.format(
                test_stats['loss'], test_stats['accuracy']))
        else:
            print('Train: Loss {} MAE {}'.format(
                train_stats['loss'], train_stats['mae']))
            print('Val: Loss {} MAE {}'.format(
                val_stats['loss'], val_stats['mae']))
            print('Test: Loss {} MAE {}'.format(
                test_stats['loss'], test_stats['mae']))
    else:
        val_stats = evaluate(model, criterion, val_loader, classification)

        train_stats = evaluate(model, criterion, train_loader, classification)

        test_stats = evaluate(model, criterion, test_loader, classification)

        print('done...')
        if classification:
            print('Train: Loss {} Acc {}'.format(
                train_stats['loss'], train_stats['accuracy']))
            print('Val: Loss {} Acc {}'.format(
                val_stats['loss'], val_stats['accuracy']))
            print('Test: Loss {} Acc {}'.format(
                test_stats['loss'], test_stats['accuracy']))
        else:
            print('Train: Loss {} MAE {}'.format(
                train_stats['loss'], train_stats['mae']))
            print('Val: Loss {} MAE {}'.format(
                val_stats['loss'], val_stats['mae']))
            print('Test: Loss {} MAE {}'.format(
                test_stats['loss'], test_stats['mae']))

if __name__ == '__main__':
    att_map = {'sex': 2, 'age': 1}
    data_map = {
        'ixi': '/vol/biomedic/users/np716/data/ixi_hh/',
        'camcan': '/vol/biomedic2/np716/data/CamCAN/'
    }
    import argparse
    parser = argparse.ArgumentParser(description='Medical Bagnet')
    parser.add_argument('--model_path', '-m',
                        default='/tmp/test_med_bag/checkpoint.pth')
    parser.add_argument('--data_type', '-d', default='ixi', choices=['ixi', 'camcan'])
    parser.add_argument('--attribute', default='sex', choices=['sex', 'age'])
    parser.add_argument('--scale', default='2mm', choices=['1mm', '2mm'])
    parser.add_argument('--rf', default=9, type=int, choices=[9, 17, 33, 177])
    parser.add_argument('--scale_factor', default=0, type=int)
    parser.add_argument('--localised', default=False, action='store_true')
    parser.add_argument('--cuda', '-c', default='0')

    args = parser.parse_args()

    if args.localised:
        args.save_path = os.path.join(
            os.path.dirname(args.model_path), 'local_{}'.format(args.attribute))
        os.makedirs(args.save_path, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    args.num_classes = att_map[args.attribute]
    args.data_path = data_map[args.data_type]

    deploy(args)
