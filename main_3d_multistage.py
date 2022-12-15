#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar
import pandas as pd

from utils.opt import Options
from utils.h36motion3d import H36motion3D

import utils.utils as utils

import utils.model_multistage as multistage
import utils.data_utils as data_utils

import random

def torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)  # Numpy module.
    np.random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(opt):
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr
    is_cuda = torch.cuda.is_available()

    # save option in log
    script_name = os.path.basename(__file__).split('.')[0]
    script_name = script_name + '_in{:d}_out{:d}'.format(opt.input_n, opt.output_n)

    # create model
    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    sample_rate = opt.sample_rate
    frame_n = input_n + output_n
    num_joint = 22
    joint_feature = 3
    joint_n = [num_joint] + opt.num_joint_list

    if opt.smoothing == 'dct':
        assert len(opt.num_hids) == (len(opt.num_dcts) + 1)

    model = multistage.final_stage(input_feature=joint_feature, num_hids=opt.num_hids, p_dropout=opt.dropout,
                                   num_blocks=opt.num_blocks, joint_n=joint_n, frame_n=frame_n, skl_type='h36m',
                                   num_pred_frame=opt.num_pred_frame, gcn_types=opt.gcn_types)

    if is_cuda:
        model.cuda()
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    # group param
    all_params = model.parameters()
    res_params = []
    for pname, p in model.named_parameters():
        if any([pname.endswith(k) for k in ['adj']]):
            res_params += [p]
            print(pname)
    params_id = list(map(id, res_params))
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    freeze_keys = [['transform', opt.freeze_epochs]]

    optimizer = torch.optim.Adam([{'params': other_params},
                                  {'params': res_params, 'lr': opt.lr/1e6}], lr=opt.lr)

    if opt.is_load:
        model_path_len = 'checkpoint/test/' + 'ckpt_' + script_name + '_last.pth.tar'
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        if is_cuda:
            ckpt = torch.load(model_path_len)
        else:
            ckpt = torch.load(model_path_len, map_location='cpu')
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    # data loading
    print(">>> loading data")
    train_dataset = H36motion3D(path_to_data=opt.data_dir, actions='all', input_n=input_n, output_n=output_n,
                                split=0, sample_rate=sample_rate, lerp_steps=opt.total_steps)
    acts = data_utils.define_actions('all')
    test_data = dict()
    for act in acts:
        test_dataset = H36motion3D(path_to_data=opt.data_dir, actions=act, input_n=input_n, output_n=output_n, split=1,
                                   sample_rate=sample_rate, num_test=opt.num_test, lerp_steps=opt.total_steps, frames_to_pred=train_dataset.frames_)
        test_data[act] = DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch,
            shuffle=False,
            num_workers=opt.job,
            pin_memory=True)
    val_dataset = H36motion3D(path_to_data=opt.data_dir, actions='all', input_n=input_n, output_n=output_n,
                              split=2, sample_rate=sample_rate, lerp_steps=opt.total_steps, frames_to_pred=train_dataset.frames_)

    # load dadasets for training
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=opt.job,
        pin_memory=True)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)
    print(">>> data loaded !")
    print(">>> train data {}".format(train_dataset.__len__()))
    print(">>> test data {}".format(test_dataset.__len__()))
    print(">>> validation data {}".format(val_dataset.__len__()))

    for epoch in range(start_epoch, opt.epochs):
        for freeze_key, freeze_epoch in freeze_keys:
            if epoch < freeze_epoch:
                for key, value in model.named_parameters():
                    if freeze_key in key:
                        value.requires_grad = False
            else:
                for key, value in model.named_parameters():
                    if freeze_key in key and not key.endswith('transform.0'):
                        value.requires_grad = True
        if (epoch + 1) % opt.lr_decay == 0:
            lr = lr_now * opt.lr_gamma
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr/1e6
            lr_now = lr

        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])

        # per epoch
        lr_now, t_l, t_l_stage1 = train(train_loader, model, optimizer, input_n=input_n, output_n=output_n, lr_now=lr_now, max_norm=opt.max_norm, is_cuda=is_cuda,
                                        dim_used=train_dataset.dim_used, opt=opt)
        ret_log = np.append(ret_log, [lr_now, t_l, t_l_stage1])
        head = np.append(head, ['lr', 't_l', 't_l_stage1'])
        v_3d, v_3d_stage1 = val(val_loader, model, is_cuda=is_cuda, dim_used=train_dataset.dim_used, opt=opt)
        ret_log = np.append(ret_log, [v_3d, v_3d_stage1])
        head = np.append(head, ['v_3d', 'v_3d_stage1'])
        test_3d_temp = np.array([])
        test_3d_head = np.array([])
        for act in acts:
            test_3d, test_3d_stage1 = test(test_data[act], model, input_n=input_n, output_n=output_n, is_cuda=is_cuda,
                                           dim_used=train_dataset.dim_used, opt=opt)
            ret_log = np.append(ret_log, test_3d)
            if output_n == 50:
                head = np.append(head, [act+'3d1400', act+'3d1520', act+'3d1680', act+'3d2000'])
            if output_n == 25:
                head = np.append(head, [act + '3d560', act + '3d1000'])
            if output_n == 10:
                head = np.append(head, [act + '3d80', act + '3d160', act + '3d320', act + '3d400'])


        ret_log = np.append(ret_log, ['-------------'])
        head = np.append(head, ['-------------'])
        ret_log = np.append(ret_log, test_3d_temp)
        head = np.append(head, test_3d_head)

        # update log file and save checkpoint
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if epoch == start_epoch:
            df.to_csv(opt.ckpt + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(opt.ckpt + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        if not np.isnan(v_3d):
            is_best = v_3d < err_best
            err_best = min(v_3d, err_best)
        else:
            is_best = False
        file_name = ['ckpt_' + script_name + '_best.pth.tar', 'ckpt_' + script_name + '_last.pth.tar']
        utils.save_ckpt({'epoch': epoch + 1,
                         'lr': lr_now,
                         'err': test_3d[0],
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        ckpt_path=opt.ckpt,
                        is_best=is_best,
                        file_name=file_name)


def smooth(data, seq_len, input_n):
    smooth_data = data.clone()
    for i in range(input_n, seq_len):
        smooth_data[:, i] = torch.mean(data[:, input_n:i+1], dim=1)
    return smooth_data


def smooth_dct(data, input_n, output_n, dct_n):
    dct_m, idct_m = data_utils.get_dct_matrix(output_n)
    dct_m = torch.from_numpy(dct_m).float().cuda()
    idct_m = torch.from_numpy(idct_m).float().cuda()
    data_in = data[:, :input_n]
    tmp = data[:, input_n:]     # (bs,25,66)
    smooth_data_dct = torch.einsum('dt,btv->bdv', dct_m[:dct_n, :], tmp)       # (bs,20,66)
    smooth_data = torch.einsum('dt,btv->bdv', idct_m[:, :dct_n], smooth_data_dct)      # (bs,25,66)
    return torch.cat([data_in, smooth_data], dim=1)


def train(train_loader, model, optimizer, input_n=None, output_n=None, lr_now=None, max_norm=True, is_cuda=False, dim_used=[], opt=None):
    t_l = utils.AccumLoss()
    t_l_stage1 = utils.AccumLoss()
    model.train()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, all_seq, pred_frame) in enumerate(train_loader):
        batch_size = inputs.shape[0]
        if batch_size == 1:
            continue
        bt = time.time()
        if is_cuda:
            inputs = inputs.cuda().float()
            all_seq = all_seq.cuda().float()
            pred_frame = pred_frame.cuda()

        output_dict = model(inputs, pred_frame)
        loss_tmp = 0
        count = 0
        if len(opt.num_hids) > 1:
            targs = []
            tmp = all_seq[:, :, dim_used]
            if opt.smoothing == 'dct':
                for num_dct in opt.num_dcts:    # [5,4,3]
                    targs.append(
                        smooth_dct(tmp, input_n, output_n, num_dct)
                    )
            if opt.smoothing == 'aas':
                for _ in range(len(opt.num_hids)-1):
                    tmp = smooth(tmp, input_n+output_n, input_n)
                    targs.append(tmp)

            for j, gcn_type in enumerate(opt.gcn_types[:-1]):
                loss_tmp = loss_tmp \
                + torch.mean(torch.norm(output_dict['y{}'.format(j)].reshape(-1, 3) - targs[-j-1].reshape(-1, 3), 2, 1))
                count += 1
                if gcn_type == 'int':
                    frame_idx = pred_frame.unsqueeze(-1).expand(-1, -1, len(dim_used))
                    loss_tmp = loss_tmp \
                    + torch.mean(torch.norm(output_dict['aux_y{}'.format(j)].reshape(-1, 3) - targs[-j-1].gather(1, frame_idx).reshape(-1, 3), 2, 1))
                    count += 1

        # calculate loss and backward
        pred = output_dict['y{}'.format(len(opt.num_hids)-1)].reshape(-1, 3)
        targ = all_seq[:, :, dim_used].reshape(-1, 3)
        loss = torch.mean(torch.norm(pred - targ, 2, 1))
        count += 1

        if opt.gcn_types[-1] == 'int':
            pre_pred = output_dict['aux_y{}'.format(len(opt.num_hids)-1)].reshape(-1, 3)
            frame_idx = pred_frame.unsqueeze(-1).expand(-1, -1, all_seq.shape[2])
            pre_targ = all_seq.gather(1, frame_idx)[:, :, dim_used].reshape(-1, 3)
            loss_stage1 = torch.mean(torch.norm(pre_pred - pre_targ, 2, 1))
            count += 1
        else:
            loss_stage1 = torch.tensor(0.)

        optimizer.zero_grad()
        total_loss = (loss_tmp + loss + loss_stage1) / count
        total_loss.backward()

        if max_norm:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()

        # update the training loss
        t_l.update(loss.cpu().data.numpy() * batch_size, batch_size)
        t_l_stage1.update(loss_stage1.cpu().data.numpy() * batch_size, batch_size)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i+1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    print('epoch time {:.2f}s'.format(time.time() - st))
    return lr_now, t_l.avg, t_l_stage1.avg


def val(train_loader, model, is_cuda=False, dim_used=[], opt=None):
    t_3d = utils.AccumLoss()
    t_3d_stage1 = utils.AccumLoss()
    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, all_seq, pred_frame) in enumerate(train_loader):
        bt = time.time()
        if is_cuda:
            inputs = inputs.cuda().float()
            all_seq = all_seq.cuda().float()
            pred_frame = pred_frame.cuda()

        with torch.no_grad():
            output_dict = model(inputs, pred_frame)
            n = inputs.shape[0]

            # pred = outputs.reshape(-1, 3)
            pred = output_dict['y{}'.format(len(opt.num_hids)-1)].reshape(-1, 3)

            targ = all_seq[:, :, dim_used].reshape(-1, 3)
            m_err = torch.mean(torch.norm(pred - targ, 2, 1))

            if opt.gcn_types[-1] == 'int':
                pre_pred = output_dict['aux_y{}'.format(len(opt.num_hids)-1)].reshape(-1, 3)

                frame_idx = pred_frame.unsqueeze(-1).expand(-1, -1, all_seq.shape[2])
                pre_targ = all_seq.gather(1, frame_idx)[:, :, dim_used].reshape(-1, 3)

                m_err_stage1 = torch.mean(torch.norm(pre_pred - pre_targ, 2, 1))
            else:
                m_err_stage1 = torch.tensor(0.)

        # update the training loss
        t_3d.update(m_err.cpu().data.numpy() * n, n)
        t_3d_stage1.update(m_err_stage1.cpu().data.numpy() * n, n)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_3d.avg, t_3d_stage1.avg


def test(train_loader, model, input_n=20, output_n=50, is_cuda=False, dim_used=[], opt=None):
    N = 0
    if output_n == 50:
        eval_frame = [34, 37, 41, 49]
    elif output_n == 25:
        eval_frame = [13, 24]
    elif output_n == 10:
        eval_frame = [1, 3, 7, 9]
    t_3d = np.zeros(len(eval_frame))

    t_3d_pred = np.zeros(opt.num_pred_frame)

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, all_seq, pred_frame) in enumerate(train_loader):
        bt = time.time()
        if is_cuda:
            inputs = inputs.cuda().float()
            all_seq = all_seq.cuda().float()
            pred_frame = pred_frame.cuda()


        with torch.no_grad():
            # joints at same loc
            joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
            index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
            joint_equal = np.array([13, 19, 22, 13, 27, 30])
            index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))
            n = inputs.shape[0]
            # _, _, _, _, _, _, outputs, pre_outputs = model(inputs)
            output_dict = model(inputs, pred_frame)
            # outputs_3d = outputs.reshape(n, input_n + output_n, -1)     # (bs,35,66)
            outputs_3d = output_dict['y{}'.format(len(opt.num_hids)-1)].reshape(n, input_n + output_n, -1)
            pred_3d = all_seq.clone()       # (bs,35,96)
            pred_3d[:, :, dim_used] = outputs_3d        # (bs,35,96)
            pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]      # (bs,35,96)
            pred_p3d = pred_3d[:, input_n:]     # (bs,25,96)
            targ_p3d = all_seq[:, input_n:]     # (bs,25,96)

            if opt.gcn_types[-1] == 'int':
                # pre_outputs_3d = pre_outputs.reshape(n, len(pred_frame), -1)     # (bs,1,22,3)->(bs,1,66)
                pre_outputs_3d = output_dict['aux_y{}'.format(len(opt.num_hids)-1)].reshape(n, opt.num_pred_frame, -1)


                # pre_pred_3d = all_seq[:, pred_frame-1].clone()  # (bs,1,96)
                frame_idx = pred_frame.unsqueeze(-1).expand(-1, -1, all_seq.shape[2])
                pre_pred_3d = all_seq.gather(1, frame_idx).clone()



                pre_pred_3d[:, :, dim_used] = pre_outputs_3d  # (bs,1,96)
                pre_pred_3d[:, :, index_to_ignore] = pre_pred_3d[:, :, index_to_equal]  # (bs,1,96)
                pre_pred_p3d = pre_pred_3d  # (bs,1,96)

                # pre_targ_p3d = all_seq[:, pred_frame-1]  # (bs,1,96)
                frame_idx = pred_frame.unsqueeze(-1).expand(-1, -1, all_seq.shape[2])
                pre_targ_p3d = all_seq.gather(1, frame_idx)


            else:
                pre_pred_p3d = torch.zeros(n, 999, 96)      # (bs,1,96)
                pre_targ_p3d = torch.zeros(n, 999, 96)      # (bs,1,96)


        for k in np.arange(0, opt.num_pred_frame):
            t_3d_pred[k] += torch.mean(torch.norm(
                pre_targ_p3d[:, k].reshape(-1, 3) - pre_pred_p3d[:, k].reshape(-1, 3), 2, 1)).cpu().data.numpy() * n


        for k in np.arange(0, len(eval_frame)):
            j = eval_frame[k]
            t_3d[k] += torch.mean(torch.norm(
                targ_p3d[:, j].reshape(-1, 3) - pred_p3d[:, j].reshape(-1, 3), 2, 1)).cpu().data.numpy() * n
        N += n
        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i+1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_3d / N, t_3d_pred / N


if __name__ == "__main__":
    torch_seed(2022)
    option = Options().parse()
    main(option)
