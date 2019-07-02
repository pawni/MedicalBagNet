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
import torch.optim as optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter

from data.ixi import IxiDataset
from data.camcan import CamCANDataset

from bagnets import bagnet9, bagnet17, bagnet33, bagnet177


def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def load_checkpoint(state, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        state.update(checkpoint)
        print('Loaded checkpoint')

        return True
    return False


def save_checkpoint(state, logdir):
    checkpoint_path = os.path.join(logdir, 'checkpoint.pth')
    best_checkpoint_path = os.path.join(logdir, 'best_checkpoint.pth')

    print('Saving checkpoint')
    torch.save(state, checkpoint_path)

    if state['is_best']:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


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
            eval_stats['mae'] = np.mean(aes)
            eval_stats['aes'] = np.array(aes).reshape((-1,))

        return eval_stats


def log_evaluation(epoch, statistics, writer, prefix):
    writer.add_scalar(prefix + '/loss', statistics['loss'], epoch)
    if 'accuracy' in statistics.keys():
        writer.add_scalar(prefix + '/accuracy', statistics['accuracy'], epoch)
        writer.add_histogram(prefix + '/probs', statistics['probs'], epoch)
        print('{} epoch {}. Loss {} Acc {}'.format(prefix, epoch, statistics['loss'], statistics['accuracy']))
    else:
        writer.add_scalar(prefix + '/mae', statistics['mae'], epoch)
        print('{} epoch {}. Loss {} MAE {}'.format(prefix, epoch, statistics['loss'], statistics['mae']))


def get_datasets(data_type, data_path, batch_size, attribute='sex', scale='2mm'):
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
            attribute=attribute, augment=True)

        val_dataset = CamCANDataset(
            os.path.join(data_path, 'val.csv'), attribute=attribute)

        test_dataset = CamCANDataset(
            os.path.join(data_path, 'test.csv'), attribute=attribute)
    else:
        raise Exception('unknown data type')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    return train_loader, val_loader, test_loader

def train(args):
    seed(args.seed)

    print('Starting run with:\n{}'.format(args))

    train_loader, val_loader, test_loader = get_datasets(
        args.data_type, args.data_path, args.batch_size, args.attribute,
        args.scale)

    if args.rf == 9:
        model = bagnet9(num_classes=args.num_classes,
                        scale_filters=args.scale_factor)
    elif args.rf == 17:
        model = bagnet17(num_classes=args.num_classes,
                         scale_filters=args.scale_factor)
    elif args.rf == 33:
        model = bagnet33(num_classes=args.num_classes,
                         scale_filters=args.scale_factor)
    elif args.rf == 177:
        model = bagnet177(num_classes=args.num_classes,
                          scale_filters=args.scale_factor)
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
        columns = ['epoch', 'eval_loss', 'eval_acc',
                   'train_loss', 'train_acc',
                   'test_loss', 'test_acc']
    elif args.attribute == 'age':
        criterion = nn.MSELoss()
        classification = False
        columns = ['epoch', 'eval_loss', 'eval_mae',
                   'train_loss', 'train_mae',
                   'test_loss', 'test_mae']
    else:
        raise Exception('attribute not known')
    print(criterion)


    stats_csv = pd.DataFrame(columns=columns)

    nn.utils.clip_grad_value_(raw_model.parameters(), 5.)
    if args.opt == 'rms':
        optimizer = torch.optim.RMSprop(
            raw_model.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.l2)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(
            raw_model.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.l2)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    state = {
        'epoch': 0,
        'step': 0,
        'state_dict': copy.deepcopy(raw_model.state_dict()),
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'scheduler': copy.deepcopy(scheduler.state_dict()),
        'best_metric': None,
        'best_epoch': 0,
        'is_best': False,
        'stats_csv': stats_csv
    }

    epochs = args.num_epochs
    patience = args.patience

    checkpoint_path = os.path.join(args.logdir, 'checkpoint.pth')
    writer = SummaryWriter(args.logdir)
    if load_checkpoint(state, checkpoint_path):
        raw_model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        stats_csv = state['stats_csv']

        if patience > 0 and state['epoch'] - state['best_epoch'] >= patience:
            print('stopped early')
            return

    save_checkpoint(state, args.logdir)

    writer.add_text('args/str', str(args), state['epoch'])
    writer.add_text('model/str', str(model), state['epoch'])

    try:
        # Train the model
        for epoch in range(state['epoch'], epochs):
            model.train()

            losses = []
            if classification:
                correct = []
            else:
                aes = []

            delayed = 0
            pbar_dict = {}
            with tqdm(train_loader, desc="Epoch [{}/{}]".format(epoch+1, epochs)) as pbar:
                for images, labels, _ in pbar:
                    if torch.cuda.is_available():
                        if torch.cuda.device_count() == 1:
                            images = images.cuda()
                        labels = labels.cuda()
                    # Forward pass
                    outputs = model(images)

                    if classification:
                        loss = criterion(outputs, labels)
                        predicted = torch.argmax(outputs.data, 1)
                    else:
                        loss = criterion(outputs.squeeze(), labels)

                    cpu_loss = loss.mean().cpu().item()

                    losses += [cpu_loss]
                    if classification:
                        np_corr = (predicted.detach() == labels).cpu().numpy()
                        correct += list(np_corr)
                    else:
                        aes += list(torch.abs(outputs.detach().squeeze()
                                              - labels).cpu().numpy())
                    # Backward and optimize
                    delayed += 1
                    if args.delayed_step > 0:
                        (loss / args.delayed_step).backward()
                    else:
                        loss.backward()

                    if args.delayed_step == 0 or (delayed + 1) % args.delayed_step == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        cur_loss = np.mean(losses)
                        writer.add_scalar('train/loss', cur_loss, state['step'])
                        if classification:
                            acc = np.mean(correct)
                            writer.add_scalar('train/accuracy', acc, state['step'])
                            correct = []

                            pbar_dict = {'loss': cur_loss, 'acc': acc}
                        else:
                            mae = np.mean(aes)
                            writer.add_scalar('train/mae', mae, state['step'])
                            aes = []

                            pbar_dict = {'loss': cur_loss, 'mae': mae * IxiDataset.age_std}
                        state['step'] += 1

                        delayed = 0
                        losses = []

                    pbar.set_postfix(**pbar_dict)

            # step last backward if the step isn't done yet because of an 'incomplete'
            # delayed / accumulated batch
            if delayed > 0:
                optimizer.step()
                optimizer.zero_grad()
                cur_loss = np.mean(losses)
                writer.add_scalar('train/loss', cur_loss, state['step'])

                if classification:
                    acc = np.mean(correct)
                    writer.add_scalar('train/accuracy', acc, state['step'])
                else:
                    mae = np.mean(aes) * IxiDataset.age_std
                    writer.add_scalar('train/mae', mae, state['step'])
                state['step'] += 1

            state['epoch'] = epoch + 1
            state['state_dict'] = copy.deepcopy(raw_model.state_dict())
            state['optimizer'] = copy.deepcopy(optimizer.state_dict())

            val_stats = evaluate(model, criterion, val_loader, classification)
            log_evaluation(epoch, val_stats, writer, 'eval')

            if state['best_metric'] is None or (
                    (not classification and state['best_metric'] > val_stats['mae']) or
                    (classification and state['best_metric'] < val_stats['accuracy'])):
                state['is_best'] = True
                if classification:
                    state['best_metric'] = val_stats['accuracy']
                else:
                    state['best_metric'] = val_stats['mae']
                state['best_epoch'] = epoch
            else:
                state['is_best'] = False

            train_stats = evaluate(model, criterion, train_loader,
                                   classification)
            log_evaluation(epoch, train_stats, writer, 'train_eval')

            test_stats = evaluate(model, criterion, test_loader, classification)
            log_evaluation(epoch, test_stats, writer, 'test')

            if classification:
                stats_csv.loc[len(stats_csv)] = [
                    epoch, val_stats['loss'], val_stats['accuracy'],
                    train_stats['loss'], train_stats['accuracy'],
                    test_stats['loss'], test_stats['accuracy'], ]
            else:
                stats_csv.loc[len(stats_csv)] = [
                    epoch, val_stats['loss'], val_stats['mae'],
                    train_stats['loss'], train_stats['mae'],
                    test_stats['loss'], test_stats['mae'], ]

            save_checkpoint(state, args.logdir)

            if patience > 0 and epoch - state['best_epoch'] >= patience:
                print('stopped early')
                writer.add_text('done/str', 'true', epoch + 1)
                break
    except RuntimeError as e:
        raise e

    print('done - stopping now')

    writer.close()

if __name__ == '__main__':
    att_map = {'sex': 2, 'age': 1}
    data_map = {
        'ixi': '/vol/biomedic/users/np716/data/ixi_hh/',
        'camcan': '/vol/biomedic2/np716/data/CamCAN/'
    }
    import argparse
    parser = argparse.ArgumentParser(description='Medical Bagnet')
    parser.add_argument('--logdir', '-l', default='/tmp/test_med_bag')
    parser.add_argument('--batch_size', '-b', default=2, type=int)
    parser.add_argument('--num_epochs', '-e', default=500, type=int)
    parser.add_argument('--patience', default=-1, type=int)
    parser.add_argument('--data_type', '-d', default='ixi', choices=['ixi', 'camcan'])
    parser.add_argument('--attribute', default='sex', choices=['sex', 'age'])
    parser.add_argument('--l2', default=0., type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--scale', default='2mm', choices=['1mm', '2mm'])
    parser.add_argument('--rf', default=9, type=int, choices=[9, 17, 33, 177])
    parser.add_argument('--delayed_step', default=16, type=int)
    parser.add_argument('--scale_factor', default=0, type=int)
    parser.add_argument('--cuda', '-c', default='0')
    parser.add_argument('--opt', default='rms', choices=['rms', 'adam'])

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    args.num_classes = att_map[args.attribute]
    args.data_path = data_map[args.data_type]

    train(args)
