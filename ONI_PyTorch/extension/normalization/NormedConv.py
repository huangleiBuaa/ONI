"""
Orthogonalization by Newton’s Iteration
"""
import torch.nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from typing import List
from torch.autograd.function import once_differentiable

__all__ = ['WN_Conv2d', 'OWN_Conv2d', 'ONI_Conv2d','ONI_ConvTranspose2d',
           'ONI_Linear']

#  norm funcitons--------------------------------


class IdentityModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(IdentityModule, self).__init__()

    def forward(self, input: torch.Tensor):
        return input

class WNorm(torch.nn.Module):
    def forward(self, weight):
        weight_ = weight.view(weight.size(0), -1)
        #std = weight_.std(dim=1, keepdim=True) + 1e-5
        norm = weight_.norm(dim=1, keepdim=True) + 1e-5
        weight_norm = weight_ / norm
        return weight_norm.view(weight.size())


class OWNNorm(torch.nn.Module):
    def __init__(self, norm_groups=1, *args, **kwargs):
        super(OWNNorm, self).__init__()
        self.norm_groups = norm_groups

    def matrix_power3(self, Input):
        B=torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)  # type: torch.Tensor
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        S = torch.matmul(Zc, Zc.transpose(1, 2))
        wm = torch.randn(S.shape).to(S)
        #Scales = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        #Us = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        for i in range(self.norm_groups):
            U, Eig, _ = S[i].svd()
            Scales = Eig.rsqrt().diag()
            wm[i] = U.mm(Scales).mm(U.t())
        W = wm.matmul(Zc)
        #print(W.matmul(W.transpose(1,2)))
        # W = oni_py.apply(weight, self.T, ctx.groups)
        return W.view_as(weight)

    def extra_repr(self):
        fmt_str = ['OWN:']
        if self.norm_groups > 1:
            fmt_str.append('groups={}'.format(self.norm_groups))
        return ', '.join(fmt_str)

class ONINorm(torch.nn.Module):
    def __init__(self, T=5, norm_groups=1, *args, **kwargs):
        super(ONINorm, self).__init__()
        self.T = T
        self.norm_groups = norm_groups
        self.eps = 1e-5

    def matrix_power3(self, Input):
        B=torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)  # type: torch.Tensor
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        S = torch.matmul(Zc, Zc.transpose(1, 2))
        eye = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        S = S + self.eps*eye
        norm_S = S.norm(p='fro', dim=(1, 2), keepdim=True)
        S = S.div(norm_S)
        B = [torch.Tensor([]) for _ in range(self.T + 1)]
        B[0] = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        for t in range(self.T):
            #B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, torch.matrix_power(B[t], 3), S)
            B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, self.matrix_power3(B[t]), S)
        W = B[self.T].matmul(Zc).div_(norm_S.sqrt())
        #print(W.matmul(W.transpose(1,2)))
        # W = oni_py.apply(weight, self.T, ctx.groups)
        return W.view_as(weight)

    def extra_repr(self):
        fmt_str = ['T={}'.format(self.T)]
        if self.norm_groups > 1:
            fmt_str.append('groups={}'.format(self.norm_groups))
        return ', '.join(fmt_str)


class ONINorm_colum(torch.nn.Module):
    def __init__(self, T=5, norm_groups=1, *args, **kwargs):
        super(ONINorm_colum, self).__init__()
        self.T = T
        self.norm_groups = norm_groups
        self.eps = 1e-5
        #print(self.eps)

    def matrix_power3(self, Input):
        B=torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)  # type: torch.Tensor
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        S = torch.matmul(Zc.transpose(1, 2), Zc)
        eye = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        S = S + self.eps*eye
        norm_S = S.norm(p='fro', dim=(1, 2), keepdim=True)
        #print(S.size())
        #S = S.div(norm_S)
        B = [torch.Tensor([]) for _ in range(self.T + 1)]
        B[0] = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        for t in range(self.T):
            #B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, torch.matrix_power(B[t], 3), S)
            B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, self.matrix_power3(B[t]), S)
        W = Zc.matmul(B[self.T]).div_(norm_S.sqrt())
        #print(W.matmul(W.transpose(1,2)))
        # W = oni_py.apply(weight, self.T, ctx.groups)
        return W.view_as(weight)

    def extra_repr(self):
        fmt_str = ['T={}'.format(self.T)]
        if self.norm_groups > 1:
            fmt_str.append('groups={}'.format(self.norm_groups))
        return ', '.join(fmt_str)


#  normedConvs--------------------------------


class WN_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 NScale=1.414, adjustScale=False, *args, **kwargs):
        super(WN_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        print('WN_Conv:---NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = WNorm()
        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out




class OWN_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 norm_groups=1, norm_channels=0, NScale=1.414, adjustScale=False, *args, **kwargs):
        super(OWN_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        if norm_channels > 0:
            norm_groups = out_channels // norm_channels

        print('OWN_Conv:----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = OWNNorm(norm_groups=norm_groups)

        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

class ONI_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 T=5, norm_groups=1, norm_channels=0, NScale=1.414, adjustScale=False, ONIRow_Fix=False, *args, **kwargs):
        super(ONI_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        print('ONI channels:--OD:',out_channels, '--ID:', in_channels, '--KS',kernel_size)
        if out_channels <= (in_channels*kernel_size*kernel_size):
            if norm_channels > 0:
                norm_groups = out_channels // norm_channels
            #print('ONI_Conv_Row:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
            self.weight_normalization = ONINorm(T=T, norm_groups=norm_groups)
        else:
            if ONIRow_Fix:
              #  print('ONI_Conv_Row:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
                self.weight_normalization = ONINorm(T=T, norm_groups=norm_groups)
            else: 
               # print('ONI_Conv_Colum:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
                self.weight_normalization = ONINorm_colum(T=T, norm_groups=norm_groups)
        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
           # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

class ONI_Linear(torch.nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True,
                 T=4, norm_groups=1, norm_channels=0, NScale=1, adjustScale=False, *args, **kwargs):
        super(ONI_Linear, self).__init__(in_channels, out_channels, bias)
        if out_channels <= in_channels:
            if norm_channels > 0:
                norm_groups = out_channels // norm_channels
            print('ONI_Linear_Row:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
            self.weight_normalization = ONINorm(T=T, norm_groups=norm_groups)
        else:
            print('ONI_Linear_Colum:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
            self.weight_normalization = ONINorm_colum(T=T, norm_groups=norm_groups)

        self.scale_ = torch.ones(out_channels, 1, ).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.linear(input_f, weight_q, self.bias)
        return out





#Trans Conv


class ONI_ConvTranspose2d(torch.nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True,
                 T=5, norm_groups=1, NScale=1.414, adjustScale=False):
        super(ONI_ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        print('ONI_Column:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = ONINorm(T=T, norm_groups=norm_groups)
        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.scale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('scale', self.scale_)

    def forward(self, input_f: torch.Tensor, output_size=None) -> torch.Tensor:
        output_padding = self._output_padding(input_f, output_size, self.stride, self.padding, self.kernel_size)
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.scale
        out = F.conv_transpose2d(input_f, weight_q, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
        return out





if __name__ == '__main__':
    SEED = 0
    torch.manual_seed(SEED)



    oni_ = ONINorm(T=5, norm_groups=1)
    w_ = torch.randn(4, 4, 2, 2)
    print(w_)
    w_.requires_grad_()
    y_ = oni_(w_)
    z_ = y_.view(w_.size(0), -1)
    #print(z_.sum(dim=1))
    print(z_.matmul(z_.t()))
#    y_.sum().backward()
#    print('w grad', w_.grad.size())
#     conv=ONI_Conv2d(4, 2, 1, adjustScale=True)
#     b = conv(w_)
#     print(b)
