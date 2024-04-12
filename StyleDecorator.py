import torch as th 
import torch.nn as nn 
import torch.nn.functional as F

class StyleDecorator(nn.Module):
    def __init__(self, style_strength= 1.0, patch_size= 3, patch_stride= 1):
        super(StyleDecorator, self).__init__()
        self.style_strength = style_strength
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def kernel_normalize(self, kernel, k= 3):
        b, ch, h, w, kk = kernel.size()
        # calc kernel norm
        kernel = kernel.view(b, ch, h*w, kk).transpose(2, 1)
        kernel_norm = th.norm(kernel.contiguous().view(b, h * w, ch * kk), p= 2, dim= 2, keepdim= True)    
        # kernel reshape
        kernel = kernel.view(b, h * w, ch, k, k)
        kernel_norm = kernel_norm.view(b, h * w, 1, 1, 1)
        return kernel, kernel_norm 

    def conv2d_with_style_kernels(self, features, kernels, patch_size, deconv_flag= False):
        output = []
        # padding
        pad = (patch_size - 1) // 2
        padding_size = (pad, pad, pad, pad)
        # batch-wise convolutions with style kernels
        for feature, kernel in zip(features, kernels):
            feature = F.pad(feature.unsqueeze(0), padding_size, 'constant', 0)
            if deconv_flag:
                padding_size = patch_size - 1
                output.append(F.conv_transpose2d(feature, kernel, padding= padding_size))
            else:
                output.append(F.conv2d(feature, kernel))   
        return th.cat(output, dim= 0) 

    def binarize_patch_score(self, features):
        outputs= []
        # batch-wise operation
        for feature in features:
            matching_indices = th.argmax(feature, dim= 0)
            one_hot_mask = th.zeros_like(feature)
            h, w = matching_indices.size()
            for i in range(h):
                for j in range(w):
                    ind = matching_indices[i, j]
                    one_hot_mask[ind, i, j] = 1
            outputs.append(one_hot_mask.unsqueeze(0)) 
        return th.cat(outputs, dim= 0)

    def norm_deconvolution(self, h, w, patch_size):
        mask = th.ones((h, w))
        fullmask = th.zeros((h + patch_size - 1, w + patch_size - 1))

        for i in range(patch_size):
            for j in range(patch_size):
                pad = (i, patch_size - i - 1, j, patch_size - j - 1)
                padded_mask = F.pad(mask, pad, 'constant', 0)
                fullmask += padded_mask

        pad_width = (patch_size - 1) // 2
        if pad_width == 0:
            deconv_norm = fullmask
        else:
            deconv_norm = fullmask[pad_width:-pad_width, pad_width:-pad_width]

        return deconv_norm.view(1, 1, h, w)
    
    def extract_patches(self, feature, patch_size, stride):
        ph, pw = patch_size
        sh, sw = stride
        # padding the feature
        padh = (ph - 1) // 2
        padw = (pw - 1) // 2
        padding_size = (padw, padw, padh, padh)
        feature = F.pad(feature, padding_size, 'constant', 0)
        # extract patches
        patches = feature.unfold(2, ph, sh).unfold(3, pw, sw)
        patches = patches.contiguous().view(*patches.size()[:-2], -1)

        return patches

    def reassemble_feature(self, normalized_content_feature, normalized_style_feature, patch_size, patch_stride):
        # get patches of style feature
        style_kernel = self.extract_patches(normalized_style_feature, [patch_size, patch_size], [patch_stride, patch_stride])
        # kernel normalize
        style_kernel, kernel_norm = self.kernel_normalize(style_kernel, patch_size)
        # convolution with style kernel(patch wise convolution)
        patch_score = self.conv2d_with_style_kernels(normalized_content_feature, style_kernel / kernel_norm, patch_size)       
        # binarization
        binarized = self.binarize_patch_score(patch_score)
        # deconv norm
        deconv_norm = self.norm_deconvolution(h= binarized.size(2), w= binarized.size(3), patch_size= patch_size)
        # deconvolution
        output = self.conv2d_with_style_kernels(binarized, style_kernel, patch_size, deconv_flag= True)
        
        return output / deconv_norm.type_as(output)

    def covsqrt_mean(self, feature, inverse= False, tolerance= 1e-14):
        b, c, _, _ = feature.size()
        mean = th.mean(feature.view(b, c, -1), dim= 2, keepdim= True)
        zeromean = feature.view(b, c, -1) - mean
        cov = th.bmm(zeromean, zeromean.transpose(1, 2))
        evals, evects = th.linalg.eigh(cov, UPLO='U')
        p = 0.5
        if inverse:
            p *= -1
        covsqrt = []
        for i in range(b):
            k = 0
            for j in range(c):
                if evals[i][j] > tolerance:
                    k = j
                    break
            covsqrt.append(th.mm(evects[i][:, k:],
                                th.mm(evals[i][k:].pow(p).diag_embed(),
                                        evects[i][:, k:].t())).unsqueeze(0))
        covsqrt = th.cat(covsqrt, dim=0)
        return covsqrt, mean    

    def whitening(self, feature):
        b, c, h, w = feature.size()
        inv_covsqrt, mean = self.covsqrt_mean(feature, inverse=True)
        normalized_feature = th.matmul(inv_covsqrt, feature.view(b, c, -1) - mean)
        return normalized_feature.view(b, c, h, w)

    def coloring(self, feature, target):
        b, c, h, w = feature.size()
        covsqrt, mean = self.covsqrt_mean(target)
        colored_feature = th.matmul(covsqrt, feature.view(b, c, -1)) + mean
        return colored_feature.view(b, c, h, w)

    def forward(self, c, s): 
        norm_c = self.whitening(c) # 1-1. content feature projection
        norm_s = self.whitening(s) # 1-2. style feature projection
        rF = self.reassemble_feature(norm_c, norm_s, # 2. swap content and style features
                                    patch_size= self.patch_size, 
                                    patch_stride= self.patch_stride)
        sF = self.coloring(rF, s) # 3. reconstruction feature with style mean and covariance matrix
        result_feature = (1 - self.style_strength) * c + self.style_strength * sF # 4. content and style interpolation
        
        return result_feature