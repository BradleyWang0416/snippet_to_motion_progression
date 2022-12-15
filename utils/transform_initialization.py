import torch

Transforms = {
    'M22to11': torch.FloatTensor([
                                    [0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0.],

                                    [0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],

                                    [0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 1/3, 0., 0.],

                                    [0., 0., 0., 0., 0., 0., 0., 0., 1/3, 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 1/3, 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1/3],

                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1/3],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1/3]
                                    ]),
    'M22to9': torch.FloatTensor([
                                    [0.25, 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0.25, 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0.25, 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0.25, 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0.25, 0., 0., 0., 0., 0., 0., 0.],

                                    [0., 0.25, 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0.25, 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0.25, 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 1., 0., 0., 0., 0., 0.],

                                    [0., 0., 0., 0., 0.5, 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.5, 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0.5, 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0.5, 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 1/3, 0., 0.],

                                    [0., 0., 0., 0., 0., 0., 1/3, 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 1/3, 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0.5, 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0.5, 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 1/3],

                                    [0., 0., 0., 0., 0., 0., 0., 0., 1/3],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 1/3]
                                    ]),
    'M22to6': torch.FloatTensor([
                                    [0.25, 0., 0., 0., 0., 0.],
                                    [0.25, 0., 0., 0., 0., 0.],
                                    [0.25, 0., 0., 0., 0., 0.],
                                    [0.25, 0., 0., 0., 0., 0.],
                                    [0., 0.25, 0., 0., 0., 0.],

                                    [0., 0.25, 0., 0., 0., 0.],
                                    [0., 0.25, 0., 0., 0., 0.],
                                    [0., 0.25, 0., 0., 0., 0.],
                                    [0., 0., 0.5, 0., 0., 0.],
                                    [0., 0., 0.5, 0., 0., 0.],

                                    [0., 0., 0., 0.5, 0., 0.],
                                    [0., 0., 0., 0.5, 0., 0.],
                                    [0., 0., 0., 0., 0.2, 0.],
                                    [0., 0., 0., 0., 0.2, 0.],
                                    [0., 0., 0., 0., 0.2, 0.],

                                    [0., 0., 0., 0., 0.2, 0.],
                                    [0., 0., 0., 0., 0.2, 0.],
                                    [0., 0., 0., 0., 0., 0.2],
                                    [0., 0., 0., 0., 0., 0.2],
                                    [0., 0., 0., 0., 0., 0.2],

                                    [0., 0., 0., 0., 0., 0.2],
                                    [0., 0., 0., 0., 0., 0.2]
                                    ]),
    'M22to2': torch.FloatTensor([
                                    [1/9, 0.],
                                    [1/9, 0.],
                                    [1/9, 0.],
                                    [1/9, 0.],
                                    [1/9, 0.],

                                    [1/9, 0.],
                                    [1/9, 0.],
                                    [1/9, 0.],
                                    [1/9, 0.],
                                    [0., 1/13],

                                    [0., 1/13],
                                    [0., 1/13],
                                    [0., 1/13],
                                    [0., 1/13],
                                    [0., 1/13],

                                    [0., 1/13],
                                    [0., 1/13],
                                    [0., 1/13],
                                    [0., 1/13],
                                    [0., 1/13],

                                    [0., 1/13],
                                    [0., 1/13]
                                    ]),
    'M25to11': torch.FloatTensor([
                                    [0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0.],

                                    [0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0.],

                                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0.],

                                    [0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.],

                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25]
                                    ]),
    'M25to2': torch.FloatTensor([
                                    [0.1, 0.],
                                    [0.1, 0.],
                                    [0.1, 0.],
                                    [0.1, 0.],
                                    [0.1, 0.],

                                    [0.1, 0.],
                                    [0.1, 0.],
                                    [0.1, 0.],
                                    [0.1, 0.],
                                    [0.1, 0.],

                                    [1/15, 0.],
                                    [1/15, 0.],
                                    [1/15, 0.],
                                    [1/15, 0.],
                                    [1/15, 0.],

                                    [1/15, 0.],
                                    [1/15, 0.],
                                    [1/15, 0.],
                                    [1/15, 0.],
                                    [1/15, 0.],

                                    [1/15, 0.],
                                    [1/15, 0.],
                                    [1/15, 0.],
                                    [1/15, 0.],
                                    [1/15, 0.],
                                    ]),
    'M22to22': torch.eye(22, dtype=torch.float32),
    'M25to25': torch.eye(25, dtype=torch.float32)
}
