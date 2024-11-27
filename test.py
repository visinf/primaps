import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.seed import seed_everything 
from pytorch_lightning import Trainer
from multiprocessing import get_context

import datasets
from primaps_modules.metrics import UnsupervisedMetrics
import primaps_modules.transforms as transforms
from primaps_modules.parser import test_parser
from primaps_modules.crf import *
from train import UnsupervisedSegmenter


class UnsupervisedSegmenter_test(UnsupervisedSegmenter):
    def __init__(self, opts):
        super(UnsupervisedSegmenter_test, self).__init__(opts)
        self.meter_theta_m = UnsupervisedMetrics("test/cluster/", self.opts.num_classes, 0, True)
        self.meter_theta_r = UnsupervisedMetrics("test/seghead/", self.opts.num_classes, 0, True)
        self.meter_linear = UnsupervisedMetrics("test/linear/", self.opts.num_classes, 0, False)
        self.meter_theta_r_crf = UnsupervisedMetrics("test/segheadcrf/", self.opts.num_classes, 0, True)
        self.meter_theta_m_crf = UnsupervisedMetrics("test/clustercrf/", self.opts.num_classes, 0, True)
        self.vis_test = []
    

    def on_test_start(self) -> None:
        super().on_test_start()
        self.meter_theta_m.to(self.device) 
        self.meter_linear.to(self.device)
        self.meter_theta_r.to(self.device)
        self.meter_theta_r_crf.to(self.device)
        self.meter_theta_m_crf.to(self.device)
        
        self.meter_theta_m.reset()
        self.meter_linear.reset()
        self.meter_theta_r.reset()
        self.meter_theta_r_crf.reset()
        self.meter_theta_m_crf.reset()


    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            # get image and label
            img, label = batch[0], batch[1]
            label[label>=self.opts.num_classes] = self.opts.ignore_label
            # backbone forwards pass with image and hfliped image
            feats = self.net(img, n=self.opts.dino_block)
            feats_flip = self.net(img.flip(dims=[3]), n=self.opts.dino_block)
            feats = (feats + feats_flip.flip(dims=[3])) / 2
            # interpolate features to label size
            feats = F.interpolate(feats, label.shape[-2:], mode='bilinear', align_corners=False)
            # get predictions
            pred = torch.log_softmax(self.linear_probe(feats.detach().clone()), dim=1)
            pred = pred.argmax(1)
            self.meter_linear.update(pred, label)

            theta_r_logits = self.seghead(feats.detach().clone())
            theta_r_prob = torch.log_softmax(theta_r_logits, dim=1)
            pred = theta_r_prob.argmax(1)
            self.meter_theta_r.update(pred, label)

            theta_m_prob = self.cluster_probe(feats.detach().clone())
            pred = theta_m_prob.argmax(1)
            self.meter_theta_m.update(pred, label)
        # apply crf
        with get_context('spawn').Pool(5) as pool:
            soft = F.log_softmax(theta_m_prob, dim=1)
            out_clstrcrf = batched_crf(pool, img, soft).argmax(1).to(self.device)
            out_shcrf = batched_crf(pool, img, theta_r_prob).argmax(1).to(self.device)
            
        self.meter_theta_m_crf.update(out_clstrcrf, label)
        self.meter_theta_r_crf.update(out_shcrf, label) 
                            

    def test_epoch_end(self, outputs) -> None:
        super().test_epoch_end(outputs)

        tb_metrics = {**self.meter_theta_m.compute(),
                      **self.meter_linear.compute(),
                      **self.meter_theta_r.compute(),
                      **self.meter_theta_m_crf.compute(),
                      **self.meter_theta_r_crf.compute()}
        
        # print results
        print('--------------------------------------------------')
        print('THETA_M      mIoU:'+ str(round(tb_metrics['test/cluster/mIoU'], 4))+'    THETA_M Acc:    '+ str(round(tb_metrics['test/cluster/Accuracy'], 4)))
        print('THETA_M CRF  mIoU:'+ str(round(tb_metrics['test/clustercrf/mIoU'], 4))+ '    THETA_M CRF Acc:    '+ str(round(tb_metrics['test/clustercrf/Accuracy'], 4)))
        print('THETA_R      mIoU:'+ str(round(tb_metrics['test/seghead/mIoU'], 4))+'    THETA_R Acc:        '+ str(round(tb_metrics['test/seghead/Accuracy'], 4)))
        print('THETA_R CRF  mIoU:'+ str(round(tb_metrics['test/segheadcrf/mIoU'], 4))+'     THETA_R CRF Acc:     '+ str(round(tb_metrics['test/segheadcrf/Accuracy'], 4)))
        print('Linear       mIoU:'+ str(round(tb_metrics['test/linear/mIoU'], 4))+'     Linear Acc:          '+ str(round(tb_metrics['test/linear/Accuracy'], 4)))
        print('--------------------------------------------------')
      
        print('Linear - THETA_M - THETA_R - THETA_M CRF - THETA_R CRF - Class Name' )
        for cls, ln, cl, sh, cltcr, shcrf in zip(self.dataset_info, tb_metrics['test/linear/Class IoUs'], tb_metrics['test/cluster/Class IoUs'], tb_metrics['test/seghead/Class IoUs'], tb_metrics['test/clustercrf/Class IoUs'], tb_metrics['test/segheadcrf/Class IoUs']):
            print(str(round(ln,2))+"; "+str(round(cl, 2))+"; "+str(round(sh, 2))+"; "+str(round(cltcr, 2))+"; "+str(round(shcrf, 2))+"; -- "+str(cls)) 

        
        
        
def main(opts):
    # set seeds
    seed_everything(seed=opts.seed, workers=True)
    # override opts in checkpoint
    dataset_root = opts.dataset_root
    checkpoint_path = opts.checkpoint_path
    print('-- Checkpoint: %s' %opts.checkpoint_path)
    opts = torch.load(opts.checkpoint_path, map_location="cpu")['hyper_parameters']['opts']
    opts.cluster_ckpt_path = ""
    opts.dataset_root = dataset_root
    opts.checkpoint_path = checkpoint_path
    opts.num_workers = 4
    opts.gpu_ids = [0]
    # load model from ckeckpoint
    model = UnsupervisedSegmenter_test(opts)
    state_dict = torch.load(checkpoint_path, map_location=model.device)['state_dict']
    if 'cluster_probe.ema_model.weight_g' in state_dict.keys():
        print('-- update state dict with ema')
        import torch.nn as nn
        model.cluster_probe = nn.utils.weight_norm(nn.Conv2d(opts.backbone_dim, opts.num_classes, (1, 1),  bias=False), name='weight', dim=0) 
        state_dict['cluster_probe.weight_g'] = state_dict['cluster_probe.ema_model.weight_g']
        state_dict['cluster_probe.weight_v'] = state_dict['cluster_probe.ema_model.weight_v']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    

    # Setup dataset
    dataset_name = os.path.split(opts.dataset_root)[-1]
    val_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.IdsToTrainIds(source=dataset_name),
                                         transforms.Resize(opts.validation_resize),     #transforms.ImgResize(opts.validation_resize),
                                         transforms.CenterCrop([opts.validation_resize[0], opts.validation_resize[0]]),
                                         transforms.Normalize()])
    val_dataset = datasets.__dict__[dataset_name](root=opts.dataset_root,
                                      split="val",
                                      transforms=val_transforms)
    val_loader = DataLoader(val_dataset,
                            batch_size=16,
                            num_workers=4,
                            sampler=None,
                            shuffle=False,
                            pin_memory=True if torch.cuda.is_available() else False)

    trainer = Trainer(benchmark = True,
                      logger=False,
                      gpus=opts.gpu_ids)
    # run test loop
    trainer.test(model, val_loader)




if __name__ == '__main__':
    opts = test_parser()
    main(opts)