#!/usr/bin/env python
# -*- coding: utf-8 -*-

import utils.model_interpolate as model_inter
import utils.model_basic_wolv2 as model_basic
import torch.nn as nn


class final_stage(nn.Module):
    def __init__(self, input_feature, num_hids, p_dropout, num_blocks, joint_n, frame_n, skl_type, num_pred_frame, gcn_types):
        super(final_stage, self).__init__()

        # self.gcn_types = ['van', 'van', 'van', 'int']
        # self.num_blocks = [1, 1, 1, 1]
        # self.num_hids = [128, 128, 128, 128]

        self.gcn_types = gcn_types
        self.num_blocks = num_blocks
        self.num_hids = num_hids

        self.stages = nn.ModuleList()

        for num_hid, num_block, gcn_type in zip(self.num_hids, self.num_blocks, self.gcn_types):
            if gcn_type == 'van':
                self.stages.append(
                    model_basic.DDGCN(input_feature=input_feature, hidden_feature=num_hid, p_dropout=p_dropout,
                                      num_block=num_block, joint_n=joint_n, frame_n=frame_n, skl_type=skl_type)
                )
            if gcn_type == 'int':
                self.stages.append(
                    model_inter.DDGCN(input_feature=input_feature, hidden_feature=num_hid, p_dropout=p_dropout,
                                      num_block=num_block, joint_n=joint_n, frame_n=frame_n, skl_type=skl_type,
                                      num_pred_frame=num_pred_frame)
                )

    def forward(self, x, pred_frame):

        output_dict = {}

        for i, gcn_type in enumerate(self.gcn_types):
            if gcn_type == 'van':
                x = self.stages[i](x)
                output_dict['y{}'.format(i)] = x
                # output_dict['aux_output_stage{}'.format(i)] = None
            if gcn_type == 'int':
                x, aux_x = self.stages[i](x, pred_frame)
                output_dict['y{}'.format(i)] = x
                output_dict['aux_y{}'.format(i)] = aux_x

        return output_dict



