from torch.utils.data import Dataset
import numpy as np
from h5py import File
import scipy.io as sio
from utils import data_utils
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd


def get_snippets(list):
    snippets = []
    for k, element in enumerate(list[:-1]):
        if len(snippets) == 0:
            snippets = np.array([[element, list[k + 1]]])
        else:
            snippets = np.concatenate((snippets, np.array([[element, list[k + 1]]])), axis=0)
    return snippets
    # 返回二维数组，比如输入list是[0,5,10,15]，则返回[[0,5],[5,10],[10,15]]


class H36motion3D(Dataset):

    def __init__(self, path_to_data, actions, input_n=20, output_n=10, split=0, sample_rate=2, num_test=-1, lerp_steps=None, frames_to_pred=None):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        subs = np.array([[1, 6, 7, 8, 9], [5], [11]])
        acts = data_utils.define_actions(actions)
        subjs = subs[split]
        all_seqs = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate, input_n + output_n, num_test)
        dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14,
                             15, 16, 17, 21, 22, 23, 24, 25, 26,
                             27, 28, 29, 30, 31, 32, 36, 37, 38,
                             39, 40, 41, 42, 43, 44, 45, 46, 47,
                             51, 52, 53, 54, 55, 56, 57, 58, 59,
                             63, 64, 65, 66, 67, 68, 75, 76, 77,
                             78, 79, 80, 81, 82, 83, 87, 88, 89,
                             90, 91, 92])
        self.all_seqs = all_seqs
        self.dim_used = dim_used
        all_seqs = all_seqs[:, :, dim_used]
        all_seqs = all_seqs.reshape(-1, input_n+output_n, len(dim_used)//3, 3)
        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)
        input_seq = all_seqs[:, i_idx]
        self.input_seq = input_seq

        if split == 0:
            pred_frame_all_samples = []
            seq_out = all_seqs[:, input_n:]  # (_,25,22,3)
            seq_out = torch.from_numpy(seq_out).permute(0, 3, 1, 2)  # (_,3,25,22)
            total_steps = lerp_steps
            for i in range(seq_out.shape[0]):
                seq_gt = seq_out[[i]]  # (1,3,10,22)
                pred_frame_one_sample = np.array([0, output_n - 1])
                step = 1
                while step <= total_steps:
                    list = get_snippets(pred_frame_one_sample)
                    for snippet in list:
                        key_poses = seq_gt[:, :, snippet]  # (1,3,2,22)
                        seq_recon = F.interpolate(key_poses, size=(snippet[1] - snippet[0] + 1, 22), mode='bilinear',
                                                  align_corners=True)  # (1,3,10,22)
                        mpjpe = torch.mean(torch.norm(seq_recon - seq_gt[:, :, snippet[0]:snippet[1] + 1, :], dim=1),
                                           dim=2)  # (1,10,22)->(1,10)
                        pred_frame = torch.argmax(mpjpe, dim=1).squeeze(0).data.numpy()  # (1)->一个数
                        pred_frame = pred_frame + snippet[0]
                        pred_frame_one_sample = np.sort(np.append(pred_frame_one_sample, pred_frame))
                    step += 1

                pred_frame_one_sample = pred_frame_one_sample[1:]

                if len(pred_frame_all_samples) == 0:
                    pred_frame_all_samples = np.expand_dims(pred_frame_one_sample, axis=0)
                else:
                    pred_frame_all_samples = np.concatenate(
                        (pred_frame_all_samples, np.expand_dims(pred_frame_one_sample, axis=0)), axis=0)
            self.frames_ = np.round(pred_frame_all_samples.mean(0) + 10)
        else:
            self.frames_ = frames_to_pred
        frames = np.expand_dims(self.frames_.astype(np.int64), axis=0)
        frames = np.repeat(frames, all_seqs.shape[0], axis=0)
        self.pred_frame = frames


    def __len__(self):
        return np.shape(self.input_seq)[0]

    def __getitem__(self, item):
        return self.input_seq[item], self.all_seqs[item], self.pred_frame[item]
