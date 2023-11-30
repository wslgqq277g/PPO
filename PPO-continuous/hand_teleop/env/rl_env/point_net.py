import torch
import torch.nn as nn
class PointNet(nn.Module): # actually pointnet
    def __init__(self, path=None,point_channel=3):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNet, self).__init__()

        print(f'PointNet')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, mlp_out_dim),
        )

        self.reset_parameters_()
        if path is not None:
            self.load_state_dict(torch.load(path))
            print('load pointnet')

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        x: [B, N, 3]
        '''
        # N = x.shape[1]
        # Local
        x = self.local_mlp(x)
        # local_feats = x
        # gloabal max pooling
        # print(x.shape,'xxxx')
        x = torch.max(x, dim=1)[0]

        return x

if __name__ == '__main__':
    pointnet = PointNet()
    c=pointnet(torch.rand(1,100,3))
    print(c.shape)