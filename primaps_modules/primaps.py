import torch
import sys
import os
import torch.nn.functional as F

sys.path.append(os.getcwd())
from primaps_modules.crf import dense_crf
from primaps_modules.median_pool import MedianPool2d


class PriMaPs():
    def __init__(self,
                 threshold=0.4,
                 ignore_id=27):
        super(PriMaPs, self).__init__()
        self.threshold = threshold
        self.ignore_id = ignore_id
        self.medianfilter = MedianPool2d(kernel_size=3, stride=1, padding=1)
    
    def _get_pseudo(self, img, feat, cls_prior):
        # initialize used pixel mask
        mask = torch.ones(feat.shape[-2:]).bool().to(feat.device)
        mask_memory = []
        pseudo_masks = []
        # get masks until 95% of features are masked or mask does not change
        while ((mask!=1).sum()/mask.numel() < 0.95):  
            _, _, v = torch.pca_lowrank(feat[:, mask].permute(1, 0), q=3, niter=100)
            # cos similarity to to c
            sim = torch.einsum("c,cij->ij", v[:, 0], F.normalize(feat, dim=0))
            # refine direction with NN
            sim[~mask] = 0
            v = F.normalize(feat, dim=0)[:, sim==sim.max()][:, 0]
            sim = torch.einsum("c,cij->ij", v, F.normalize(feat, dim=0))
            sim[~mask] = 0
            # apply threshhold and norm
            sim[sim<self.threshold*sim.max()] = 0
            sim = sim/sim.max()
            pseudo_masks.append(sim.clone())
            # update mask
            mask[sim>0]=0 
            mask_memory.insert(0, mask.clone())
            if mask_memory.__len__() > 3:
                mask_memory.pop()
                if torch.Tensor([(mask_memory[0]==i).all() for i in mask_memory]).all(): 
                    break
        # insert bg mask and stack
        pseudo_masks = (self.medianfilter(torch.stack(pseudo_masks, dim=0).unsqueeze(0)).squeeze()*10).clamp(0, 1)
        bg = (torch.mean(pseudo_masks[pseudo_masks!=0])*torch.ones(feat.shape[-2:], device=feat.device)-pseudo_masks.sum(dim=0)).unsqueeze(0).clamp(0, 1)  
        
        if (pseudo_masks.shape).__len__() == 2:
            pseudo_masks = pseudo_masks.unsqueeze(0)
        pseudo_masks = torch.cat([bg, pseudo_masks], dim=0)
        pseudo_masks = F.log_softmax(pseudo_masks, dim=0)
        # apply crf to refine masks
        pseudo_masks = dense_crf(img.squeeze(), pseudo_masks).argmax(0)
        pseudo_masks = torch.Tensor(pseudo_masks).to(feat.device)

        if (cls_prior == 0).all():
            pseudolabel = pseudo_masks
            pseudolabel[pseudolabel==0] = self.ignore_id
        else:
            pseudolabel = torch.ones(img.shape[-2:]).to(feat.device)*self.ignore_id
            for i in pseudo_masks.unique()[pseudo_masks.unique()!=0]:
                # only look at not assigned and attended pixels
                mask = (pseudolabel==self.ignore_id)*(pseudo_masks==i)
                pseudolabel[mask] = int(torch.mode(cls_prior[mask])[0])        
        return pseudolabel
    
    # multiprocessing wrapper
    def _apply_batched_decompose(self, tup):
        return self._get_pseudo(tup[0], tup[1], tup[2])

    def __call__(self, pool, imgs, features, cls_prior): 
        outs = pool.map(self._apply_batched_decompose, zip(imgs, features, cls_prior))
        return torch.stack(outs, dim=0)