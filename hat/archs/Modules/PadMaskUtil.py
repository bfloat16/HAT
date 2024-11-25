import torch

class PadMaskUtil():
    def __init__(self,model_norm_type='pre', use_pad_mask=False, pad_token=0):
        self.use_pad_mask=use_pad_mask
        self.pad_token=pad_token
        self.model_norm_type=model_norm_type

    def apply_pad_mask(self, x,mask=None):
        if self.model_norm_type == 'ppn':
            if self.use_pad_mask and mask is not None:
                x,res=x

                x = torch.masked_fill(x, mask.unsqueeze(-1).logical_not(), 0)

                res = torch.masked_fill(res, mask.unsqueeze(-1).logical_not(), 0)
                x = (x, res)
        elif self.model_norm_type == 'hc':
            if self.use_pad_mask and mask is not None:
                x = torch.masked_fill(x, mask.unsqueeze(-1).unsqueeze(-1).logical_not(), 0)
        else:
            if self.use_pad_mask and mask is not None:
                x=torch.masked_fill(x, mask.unsqueeze(-1).logical_not(), self.pad_token)
        return x