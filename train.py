import os
import sys
import random
import pathlib
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm as tqdm_bar
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.seed import seed_everything 
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from multiprocessing import get_context
torch.autograd.set_detect_anomaly(True)


from primaps_modules.primaps import PriMaPs
from primaps_modules.backbone.dino.dinovit import DinoFeaturizerv2
from primaps_modules.ema import EMA
import datasets
from datasets.cocostuff import get_coco_labeldata
from primaps_modules.metrics import UnsupervisedMetrics
import primaps_modules.transforms as transforms
from primaps_modules.visualization import visualize_confusion_matrix, batch_visualize_segmentation
from datasets.cityscapes import classes as cs_classes
from datasets.potsdam import classes as pd_classes
from primaps_modules.parser import train_parser
from primaps_modules.clustering import ConvClusterProbe
from primaps_modules.clustering import ClusterLoss


    

    
    

class UnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        if os.path.split(self.opts.dataset_root)[-1] == 'cityscapes':
            self.dataset_info = cs_classes
        elif os.path.split(self.opts.dataset_root)[-1] == 'cocostuff':
            self.dataset_info = get_coco_labeldata()[0]
        elif os.path.split(self.opts.dataset_root)[-1] == 'potsdam':
            self.dataset_info = pd_classes
            
        if ('stego_ckpt' in [a[0] for a in vars(opts).items()] and opts.stego_ckpt != ''):
            print('-- train on top of stego')
            sys.path.append(os.path.join(os.getcwd(), 'external', 'STEGO', 'src'))
            from train_segmentation import LitUnsupervisedSegmenter
            
            class STEGOFeaturizer():
                def __init__(self, stego_ckpt):
                    super(STEGOFeaturizer).__init__()
                    self.model = LitUnsupervisedSegmenter.load_from_checkpoint(stego_ckpt).net.to('cuda:'+str(opts.gpu_ids[0]))
                    opts.backbone_dim = self.model.cluster1[0].weight.shape[0]
                def __call__(self, x, n):
                    return self.model(x)[-1]
            self.net = STEGOFeaturizer(opts.stego_ckpt)
        elif ('hp_ckpt' in [a[0] for a in vars(opts).items()] and opts.hp_ckpt != ''):
            print('-- train on top of HP')
            sys.path.append(os.path.join(os.getcwd(), 'external', 'HP'))
            from build import build_model
            from utils.common_utils import parse

            class HPFeaturizer():
                def __init__(self, ckpt_dir, opt_dir):
                    super(HPFeaturizer).__init__()
                    parser_opt = parse(opts.hp_opt)
                    self.model, _, _ = build_model(opt=parser_opt["model"],
                                                   n_classes=opts.num_classes,
                                                   is_direct=parser_opt["eval"]["is_direct"])
                    self.model.to(torch.device('cuda:'+str(opts.gpu_ids[0])))
                    self.model.eval()
                    checkpoint_loaded = torch.load(ckpt_dir, map_location=torch.device('cuda:'+str(opts.gpu_ids[0])))
                    self.model.load_state_dict(checkpoint_loaded['net_model_state_dict'], strict=True)
                    opts.backbone_dim = self.model.ema_model1[0].weight.shape[0]
                def __call__(self, x, n):
                    return self.model(x)[1]
            self.net = HPFeaturizer(opts.hp_ckpt, opts.hp_opt)
            
        else:
            self.net = DinoFeaturizerv2(self.opts.backbone_arch, self.opts.backbone_patch)
                      
        self.cluster_probe = ConvClusterProbe(opts.backbone_dim, opts.num_classes)
        self.cluster_loss = ClusterLoss()
        self.test_cluster_metrics = UnsupervisedMetrics("final/cluster/", self.opts.num_classes, 0, True)

        self.pseudo_decomposer = PriMaPs(threshold=opts.threshold,
                                                 ignore_id=opts.ignore_label)

        self.linear_probe = nn.Conv2d(opts.backbone_dim, self.opts.num_classes, (1, 1))
        self.linear_probe_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.opts.ignore_label, reduction='none')
        self.test_linear_metrics = UnsupervisedMetrics("final/linear/", self.opts.num_classes, 0, False)

        self.seghead = nn.utils.weight_norm(nn.Conv2d(opts.backbone_dim, self.opts.num_classes, (1, 1),  bias=False), name='weight', dim=0) 
                    
        class FocalLoss():
            def __init__(self, gamma, ignore_id):
                super(FocalLoss).__init__()
                self.gamma = gamma
                self.ignore_index = ignore_id
                
            def __call__(self, logits, pseudo_gt):
                mean_cls_conf = logits.detach().clone().softmax(1).permute(1, 0, 2, 3).reshape(logits.size(1), -1).mean(dim=1)
                focal_weight = (1-mean_cls_conf)**self.gamma
                focalloss = F.cross_entropy(logits, pseudo_gt, weight=focal_weight ,ignore_index=self.ignore_index, reduction="none").mean()
                return focalloss
        
        self.seghead_loss = FocalLoss(opts.seghead_focalloss, opts.ignore_label)
        self.seghead_metrics = UnsupervisedMetrics("final/seghead/", self.opts.num_classes, 0, True)
        
        self.max_metrics = {'THETA_M Accuracy': 0, 'Linear Accuracy': 0, 'THETA_R Accuracy': 0,
                            'THETA_M mIoU': 0, 'Linear mIoU': 0, 'THETA_R mIoU': 0}

        self.automatic_optimization = False
        self.vis_img_eval = []
        self.vis_train = []
        self.save_hyperparameters()

        if self.opts.cluster_ckpt_path != "":
            print('-- Init cluster from checkpoint...')
            weights = torch.load(self.opts.cluster_ckpt_path)['state_dict']['cluster_probe.cluster_centers.weight_v'].to(self.device)
            self.cluster_probe.cluster_centers.weight_v = torch.nn.Parameter(weights.detach().clone())
            self.seghead.weight_v = torch.nn.Parameter(weights.detach().clone())
            self.seghead.weight_g = torch.nn.Parameter(torch.ones_like(self.seghead.weight_g))
            
        if ('stego_ckpt' in [a[0] for a in vars(opts).items()] and opts.stego_ckpt != ''):
            print('-- cluster weights from STEGO')
            cluster_weights = LitUnsupervisedSegmenter.load_from_checkpoint(opts.stego_ckpt).cluster_probe.clusters
            self.cluster_probe.cluster_centers.weight_v = torch.nn.Parameter(cluster_weights.unsqueeze(-1).unsqueeze(-1).detach().clone())
            self.seghead.weight_v = torch.nn.Parameter(cluster_weights.unsqueeze(-1).unsqueeze(-1).detach().clone())
            self.seghead.weight_g = torch.nn.Parameter(torch.ones_like(self.seghead.weight_g))
            
        if ('hp_ckpt' in [a[0] for a in vars(opts).items()] and opts.hp_ckpt != ''):
            print('-- cluster weights from HP')
            cluster_weights = torch.load(opts.hp_ckpt)['cluster_model_state_dict']['clusters']
            self.seghead.weight_v = torch.nn.Parameter(cluster_weights.unsqueeze(-1).unsqueeze(-1).detach().clone())
            self.seghead.weight_g = torch.nn.Parameter(torch.ones_like(self.seghead.weight_g))
            

    def training_step(self, batch, batch_idx):
        log_args = dict(sync_dist=False, rank_zero_only=True)
        
        # Optimizers
        linear_probe_optim, cluster_probe_optim, seghead_optim = self.optimizers()
        linear_probe_optim.zero_grad()
        cluster_probe_optim.zero_grad()
        seghead_optim.zero_grad()

        label = batch[1]
        with torch.no_grad():
            features = self.net(batch[0], n=self.opts.dino_block).detach().clone()

        lin_feats = features.detach().clone()
        linear_logits = self.linear_probe(lin_feats)           
        linear_logits = F.interpolate(linear_logits, batch[1].shape[-2:], mode='bilinear', align_corners=False)
        linear_loss = self.linear_probe_loss_fn(linear_logits, label.squeeze(1)).mean()
        
               
        if self.opts.train_state == 'baseline':
            cluster_probs = self.cluster_probe(features.detach().clone())
            cluster_loss, _ = self.cluster_loss(cluster_probs)
            cluster_preds = cluster_probs.argmax(1) 
            seghead_logits = self.seghead(features.detach().clone())
            pseudogt = cluster_preds
            seghead_loss = 0.0
            
            
        if self.opts.train_state == 'method':  
            cluster_loss = 0.0 
            features = F.interpolate(features, label.shape[-2:], mode='bilinear', align_corners=False)
            cluster_probs = self.cluster_probe(features)
            cluster_preds = cluster_probs.argmax(1)   

            if (self.opts.precomp_primaps or self.opts.precomp_primaps_root != ''):
                cls_prior = cluster_preds
                    
                pseudogt = []                
                for cls, msk in zip(cls_prior, batch[-1].squeeze()):
                    pseudo = torch.ones_like(msk).to('cpu')*self.opts.ignore_label
                    for m in msk.unique()[msk.unique()!=self.opts.ignore_label]:
                        fill = (pseudo==self.opts.ignore_label)*(msk==m).to('cpu')
                        try:
                            pseudo_id = cls[fill][cls[fill] != self.opts.ignore_label].mode()[0]
                        except:
                            print('-- Error matching pseudo mask')
                            pseudo_id = self.opts.ignore_label
                        pseudo[fill] = pseudo_id
                    pseudogt.append(pseudo)
                pseudogt = torch.stack(pseudogt).long().to(self.device)
                    
            if self.opts.student_augs:
                with torch.no_grad():
                    features = self.net(batch[2], n=self.opts.dino_block).detach().clone()
                    features = F.interpolate(features, label.shape[-2:], mode='bilinear', align_corners=False)
                
            seghead_logits = self.seghead(features.detach().clone())
            seghead_loss = self.seghead_loss(seghead_logits, pseudogt)

        loss = linear_loss + cluster_loss + seghead_loss
        self.log('loss/linear', linear_loss, **log_args)
        self.log('loss/cluster', cluster_loss, **log_args)
        self.log('loss/seghead', seghead_loss, **log_args)

        self.manual_backward(loss)
        linear_probe_optim.step()
        cluster_probe_optim.step()
        seghead_optim.step()
        
        ### Exponential Moving Average
        if self.opts.train_state == 'method':
            if self.global_step > 0 and self.global_step%self.opts.ema_update_step == 0 and self.opts.ema_update_step != -1: 
                print('-- EMA weights to cluster')
                self.cluster_probe.update()
    

        ### Visualize 
        if batch_idx == 0 and self.opts.gpu_ids.__len__() == 1:
            print('-- Visualize Train Batch')
            num_vis = 4
            cluster_preds = F.interpolate(cluster_preds.unsqueeze(1).float(), label.shape[-2:], mode='nearest').long().squeeze(1).to('cpu')
            seghead_logits = F.interpolate(seghead_logits.float(), label.shape[-2:], mode='nearest').long().to('cpu')
            pseudogt = F.interpolate(pseudogt.unsqueeze(1).float(), label.shape[-2:], mode='nearest').long().squeeze(1).to('cpu')
            self.test_cluster_metrics.update(cluster_preds.to('cpu'), label.to('cpu'))
            self.seghead_metrics.update(seghead_logits.argmax(1).to('cpu'), label.to('cpu'))  
            self.test_cluster_metrics.compute()  
            self.seghead_metrics.compute()
            vis_pseudo = pseudogt.detach().clone().to('cpu')
            vis_pseudo[vis_pseudo<self.opts.ignore_label] = self.test_cluster_metrics.map_clusters(vis_pseudo[vis_pseudo<self.opts.ignore_label]).to('cpu')
            label[label==self.opts.ignore_label] = self.opts.num_classes
            vis_pseudo[vis_pseudo==self.opts.ignore_label] = self.opts.num_classes
            self.vis_train = batch_visualize_segmentation(batch[0][:num_vis].to('cpu'),
                                                        label[:num_vis].to('cpu'),
                                                        in1=['Linear', linear_logits[:num_vis].argmax(1).detach().clone().to('cpu')],
                                                        in2=['Cluster', self.test_cluster_metrics.map_clusters(cluster_preds[:num_vis].detach().clone().to('cpu'))],
                                                        in3=['Pseudo', vis_pseudo[:num_vis].to('cpu')],
                                                        in4=['Seg Head', self.seghead_metrics.map_clusters(seghead_logits[:num_vis].argmax(1).detach().clone().to('cpu'))],
                                                        dataset_name=os.path.split(self.opts.dataset_root)[-1]).transpose(2, 0, 1)
            self.test_cluster_metrics.reset()
            self.seghead_metrics.reset()
            
        return loss


    def on_train_start(self):
        if self.opts.train_state == 'method':
            print('-- Init EMA')
            self.cluster_probe = EMA(self.seghead, beta=self.opts.ema_decay, update_after_step=0, update_every=1)


    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            img, label = batch[0], batch[1]
            feats_small = self.net(img, n=self.opts.dino_block)
            feats = F.interpolate(feats_small.clone(), label.shape[-2:], mode='bilinear', align_corners=False)
            pred = self.linear_probe(feats.detach().clone()).argmax(1)
            self.test_linear_metrics.update(pred.cpu(), label.cpu())
            linear_preds = pred.cpu()

            seghead_logits = self.seghead(feats.detach().clone())
            pred = seghead_logits.argmax(1)
            self.seghead_metrics.update(pred.cpu(), label.cpu())
            seghead_preds = pred.cpu()          

            pred = self.cluster_probe(feats.detach().clone())
            pred = pred.argmax(1)
            self.test_cluster_metrics.update(pred.cpu(), label.cpu())
            cluster_preds = pred.cpu()

            if self.trainer.num_val_batches[0]//3 <= batch_idx <= (self.trainer.num_val_batches[0]//3)+3:
                if self.trainer.num_val_batches[0]//3 == batch_idx:
                    self.vis_img_eval = []
                label[label==self.opts.ignore_label] = self.opts.num_classes
                self.vis_img_eval.append([img.to('cpu'), label.to('cpu'), linear_preds.to('cpu'), cluster_preds.to('cpu'), seghead_preds.to('cpu')])
                if batch_idx == (self.trainer.num_val_batches[0]//3)+3: 
                    self.test_cluster_metrics.compute()
                    self.seghead_metrics.compute()
                    self.vis_img_eval = batch_visualize_segmentation(torch.cat([i[0] for i in self.vis_img_eval], dim=0),
                                                                    torch.cat([i[1] for i in self.vis_img_eval], dim=0),
                                                                    in1=['Linear',torch.cat([i[2] for i in self.vis_img_eval], dim=0)],
                                                                    in2=['Cluster', self.test_cluster_metrics.map_clusters(torch.cat([i[3] for i in self.vis_img_eval], dim=0))],
                                                                    in3=['Seg Head', self.seghead_metrics.map_clusters(torch.cat([i[4] for i in self.vis_img_eval], dim=0))],
                                                                    dataset_name=os.path.split(self.opts.dataset_root)[-1]).transpose(2, 0, 1)

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        log_args = dict(sync_dist=False, rank_zero_only=True)
        tb_metrics = {
            **self.test_linear_metrics.compute(),
            **self.test_cluster_metrics.compute(),
            **self.seghead_metrics.compute(),
        }
        print(tb_metrics)

        print('--------------------------------------------------')
        print('THETA_M mIoU: '+ str(round(tb_metrics['final/cluster/mIoU'], 4)))
        print('THETA_R mIoU: '+ str(round(tb_metrics['final/seghead/mIoU'], 4)))
        print('Linear  mIoU: '+ str(round(tb_metrics['final/linear/mIoU'], 4)))
        print('--------------------------------------------------')
        log_args = dict(sync_dist=False, rank_zero_only=True)
        self.log('iou/linear', tb_metrics['final/linear/mIoU'], **log_args)
        self.log('accuracy/linear', tb_metrics['final/linear/Accuracy'], **log_args)
        self.log('iou/cluster', tb_metrics['final/cluster/mIoU'], **log_args)
        self.log('accuracy/cluster', tb_metrics['final/cluster/Accuracy'], **log_args)
        self.log('iou/seghead', tb_metrics['final/seghead/mIoU'], **log_args)
        self.log('accuracy/seghead', tb_metrics['final/seghead/Accuracy'], **log_args)

        if self.max_metrics['THETA_M mIoU'] < tb_metrics['final/cluster/mIoU']:
            self.max_metrics['THETA_M mIoU'] = tb_metrics['final/cluster/mIoU']
            self.max_metrics['THETA_M Accuracy'] = tb_metrics['final/cluster/Accuracy']
        if self.max_metrics['Linear mIoU'] < tb_metrics['final/linear/mIoU']:
            self.max_metrics['Linear mIoU'] = tb_metrics['final/linear/mIoU']
            self.max_metrics['Linear Accuracy'] = tb_metrics['final/linear/Accuracy']
        if self.max_metrics['THETA_R mIoU'] < tb_metrics['final/seghead/mIoU']:
            self.max_metrics['THETA_R mIoU'] = tb_metrics['final/seghead/mIoU']
            self.max_metrics['THETA_R Accuracy'] = tb_metrics['final/seghead/Accuracy']
        max_metrics_txt = "".join("\t" + line for line in json.dumps([str(met)+" : "+str(self.max_metrics[met]) for met in self.max_metrics], indent=2).splitlines(True))
        self.logger.experiment.add_text('Max Matrics', max_metrics_txt, self.global_step)
        print('-- Max Metrics: '+str(self.max_metrics))

        clsiou_txt = "".join("\t" + line for line in json.dumps([str(round(ln,2))+"; "+str(round(cl, 2))+"; "+str(round(sh, 2))+"; -- "+str(cls) for cls, ln, cl, sh in zip(self.dataset_info, tb_metrics['final/linear/Class IoUs'], tb_metrics['final/cluster/Class IoUs'], tb_metrics['final/seghead/Class IoUs'])], indent=2).splitlines(True))
        self.logger.experiment.add_text('Class IoUs', clsiou_txt, global_step=self.global_step)
        self.logger.experiment.add_image('Cluster Confusion Matrix', 
                                        visualize_confusion_matrix(self.dataset_info, self.test_cluster_metrics.to('cpu')).transpose(2, 0, 1), 
                                        global_step=self.global_step)
        if self.vis_img_eval != []:
            self.logger.experiment.add_image('Eval Visualization', self.vis_img_eval, global_step=self.global_step)
        if self.vis_train != []:
            self.logger.experiment.add_image('Train Visualization', self.vis_train, global_step=self.global_step)
        self.logger.experiment.add_image('Linear Confusion Matrix', 
                                        visualize_confusion_matrix(self.dataset_info, self.test_linear_metrics.to('cpu')).transpose(2, 0, 1), 
                                        global_step=self.global_step)  
        self.test_linear_metrics.reset()
        self.test_cluster_metrics.reset()
        self.seghead_metrics.reset()


    def configure_optimizers(self):
        linear_probe_optim = torch.optim.Adam(list(self.linear_probe.parameters()), lr=self.opts.linear_lr)
        seghead_optim = torch.optim.Adam(list(self.seghead.parameters()), lr=self.opts.seghead_lr)
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=self.opts.cluster_lr) 
        return linear_probe_optim, cluster_probe_optim, seghead_optim
        
        


def main(opts):
    # Create checkpoints directory
    now = datetime.now().strftime("%y%m%d-%H%M%S")
    opts.checkpoints_root = os.path.join(opts.checkpoints_root, now.split('-')[0])
    opts.log_name = opts.log_name+'_'+now.split('-')[1]
    opts.logg_root = os.path.join(opts.logg_root, now.split('-')[0])
    pathlib.Path(opts.checkpoints_root).mkdir(parents=True, exist_ok=True)
    pathlib.Path(opts.logg_root).mkdir(parents=True, exist_ok=True)
    # Setup dataset
    seed_everything(seed=opts.seed, workers=True)
    np.random.seed(opts.seed)
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(opts.seed)

    dataset_name = os.path.split(opts.dataset_root)[-1]
    train_transforms = transforms.Compose([transforms.Resize([opts.crop_size[0]]),
                                           transforms.CenterCrop([opts.crop_size[0], opts.crop_size[0]]),
                                           transforms.RandomResizedCrop(opts.crop_size, opts.augs_randcrop_scale, opts.augs_randcrop_ratio),
                                           transforms.ToTensor(),
                                           transforms.IdsToTrainIds(source=dataset_name),
                                           transforms.Normalize()])
    val_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.IdsToTrainIds(source=dataset_name),
                                         transforms.Resize(opts.validation_resize), 
                                         transforms.CenterCrop([opts.validation_resize[0], opts.validation_resize[0]]),
                                         transforms.Normalize()])

    train_dataset = datasets.__dict__[dataset_name](root=opts.dataset_root,
                                        split="train",
                                        transforms=train_transforms)
    val_dataset = datasets.__dict__[dataset_name](root=opts.dataset_root,
                                      split="val",
                                      transforms=val_transforms)

    # Dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=opts.batch_size,
                              num_workers=opts.num_workers,
                              sampler=None,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset,
                            batch_size=opts.eval_batch_size,
                            num_workers=opts.num_workers,
                            sampler=None,
                            shuffle=False,
                            pin_memory=True if torch.cuda.is_available() else False)
    model = UnsupervisedSegmenter(opts)
    print('-- Dataset sizes: Train: '+str(train_dataset.__len__())+' Val: '+str(val_dataset.__len__()))

 
    if opts.precomp_primaps:
        train_transforms = transforms.Compose([transforms.Resize([opts.crop_size[0]]),
                                               transforms.CenterCrop([opts.crop_size[0], opts.crop_size[0]]),
                                               transforms.RandomResizedCrop(opts.crop_size, opts.augs_randcrop_scale, opts.augs_randcrop_ratio),
                                               transforms.ToTensor(),
                                               transforms.IdsToTrainIds(source=dataset_name),
                                               transforms.Normalize()])
        train_dataset = datasets.__dict__[dataset_name](root=opts.dataset_root,
                                                        split="train",
                                                        transforms=train_transforms)
        train_loader = DataLoader(train_dataset,
                                  batch_size=opts.batch_size,
                                  num_workers=opts.num_workers,
                                  sampler=None,
                                  shuffle=False,
                                  pin_memory=True if torch.cuda.is_available() else False)


        print('-- Precompute Pseudo Dataset')
        dir_names = ['imgs', 'gts', 'pseudos']
        if opts.precomp_primaps_root != '':
            root_pth = opts.precomp_primaps_root
        else:
            root_pth = os.path.join(os.path.dirname(opts.dataset_root), 'cached_datasets', opts.log_name)
            for dir_name in dir_names:
                os.makedirs(os.path.join(root_pth, dir_name), exist_ok=True)
        model.to(torch.device('cuda', opts.gpu_ids[0]))

        # Write parser to folder
        f = open(os.path.join(root_pth, 'parser.txt'), "a")
        for a, b in vars(opts).items():
            f.write(str(a)+" : "+str(b)+"\n")
        f.close()
        
        
        with get_context('forkserver').Pool(opts.num_workers) as pool:
            for _, (batch) in tqdm_bar(enumerate(train_loader)):
                
                names = [os.path.split(p)[-1].replace('_leftImg8bit.png', '').replace('.jpg', '')+'.png' for p in batch[-1]]
                mask = torch.Tensor([os.path.isfile(os.path.join(root_pth, dir_names[-1], i))==False for i in names]).bool()
                if (mask==False).all(): 
                    print('-- All pseudos already computed')
                    continue
                else:
                    batch[0] = batch[0][mask]
                    batch[1] = batch[1][mask]
                    names = [n for i, n in enumerate(names) if mask[i]==True]   

                print('-- Compute %s' %batch[-1])
                imgs = batch[0].to(model.device)
                labels = batch[1]  
                with torch.no_grad():
                    feats = model.net(imgs, n=opts.dino_block)
                    pseudos = model.pseudo_decomposer(pool, batch[0], feats, torch.zeros_like(feats)).long()
                for name, i ,l, p in zip(names, imgs.to('cpu'), labels, pseudos.to('cpu')):
                    VF.to_pil_image(transforms.UnNormalize()(i)).save(os.path.join(root_pth, dir_names[0], name))
                    torchvision.io.write_png(l.type(torch.uint8), os.path.join(root_pth, dir_names[1], name))
                    print('-- Write data to %s', os.path.join(root_pth, dir_names[2], name))
                    torchvision.io.write_png(p.unsqueeze(0).type(torch.uint8), os.path.join(root_pth, dir_names[2], name))
                del pseudos, feats, imgs, labels
                torch.cuda.empty_cache()
            print('-- Cached dataset successfully')
            sys.exit()

    if (opts.precomp_primaps or opts.precomp_primaps_root != ''):
        train_transforms = [
            transforms.Resize([opts.crop_size[0]]),
            transforms.CenterCrop([opts.validation_resize[0], opts.validation_resize[0]]),
            transforms.RandomResizedCrop(opts.crop_size, opts.augs_randcrop_scale, opts.augs_randcrop_ratio),
            transforms.RandomHFlip()]
        if opts.augs_photometric: train_transforms + [
            transforms.RandGaussianBlur(), 
            transforms.ColorJitter(), 
            transforms.MaskGrayscale()]
        train_transforms = transforms.Compose(train_transforms+[transforms.ToTensor(), transforms.Normalize()], opts.student_augs)
        train_dataset = datasets.PrecomputedDataset(opts.precomp_primaps_root if opts.precomp_primaps_root != '' else root_pth,
                                                    transforms=train_transforms,
                                                    student_augs=opts.student_augs)
        train_loader = DataLoader(train_dataset,
                                  batch_size=opts.batch_size,
                                  num_workers=opts.num_workers,
                                  sampler=None,
                                  shuffle=True,
                                  drop_last=True,
                                  pin_memory=True if torch.cuda.is_available() else False)
        
        print('-- Cached Dataset sizes: Train: '+str(train_dataset.__len__())+' Val: '+str(val_dataset.__len__()))
        print('-- Test cached dataset')
        try:
            all_iter = 0
            for batch in tqdm_bar(train_loader):
                count_iter = batch[-1].clone()
                count_iter[count_iter==255] = 0
                all_iter = all_iter + count_iter.squeeze(1).flatten(1).max(dim=-1)[0].sum()
            print('Iter per sample %s' %(all_iter/train_loader.sampler.num_samples))
        except:
            print('-- Cached dataset corrupted!')
            sys.exit()  
    
    if opts.pcainit:
        print('-- PCA cluster initialization')
        with torch.no_grad():
            model.to(torch.device('cuda', opts.gpu_ids[0]))
            pca_init = []
            for idx, (batch) in tqdm_bar(enumerate(train_loader)): 
                if idx > opts.num_batch_pcainit: 
                    break 
                feats = model.net(batch[0].to(device=model.device), n=opts.dino_block)
                pca_init.append(feats.permute(1, 0, 2, 3).reshape(feats.size(1), -1).to('cpu'))
            model.to('cpu')
            pca_init = torch.cat(pca_init, 1).permute(1, 0).to(torch.device('cuda', opts.gpu_ids[0]))

        _, _, v = torch.pca_lowrank(pca_init, q=2*opts.num_classes, niter=200)
        v = v[:, :opts.num_classes] 
        pca_weights = v.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
        model.cluster_probe.cluster_centers.weight_v = torch.nn.Parameter(pca_weights.detach().clone())
        model.cluster_probe.cluster_centers.weight_g = torch.nn.Parameter(torch.ones_like(model.cluster_probe.cluster_centers.weight_g))
        model.seghead.weight_v = torch.nn.Parameter(pca_weights.detach().clone())
        model.seghead.weight_g = torch.nn.Parameter(torch.ones_like(model.seghead.weight_g))
        del feats, batch, pca_init, v, pca_weights
        torch.cuda.empty_cache()

    if dataset_name == 'cityscapes':
        gpu_args = dict(gpus=opts.gpu_ids, check_val_every_n_epoch=opts.validation_freq, num_sanity_val_steps=-1)
    elif dataset_name == 'cocostuff':
        gpu_args = dict(gpus=opts.gpu_ids, val_check_interval=100, num_sanity_val_steps=-1)
    elif dataset_name == 'potsdam':
        gpu_args = dict(gpus=opts.gpu_ids, check_val_every_n_epoch=opts.validation_freq, num_sanity_val_steps=-1)
    
    if len(opts.gpu_ids) > 1:
        gpu_args['accelerator'] = 'ddp'
    logger = TensorBoardLogger(os.path.join(opts.logg_root, opts.log_name), default_hp_metric=False)
    parser_txt = "".join("\t" + line for line in json.dumps([str(a)+" : "+str(b) for a, b in vars(opts).items()], indent=2).splitlines(True))
    logger.experiment.add_text('Parser', parser_txt)
        
    trainer = Trainer(max_epochs=opts.num_epochs,
                      log_every_n_steps=1,
                      logger=logger,
                      benchmark = True,
                      enable_checkpointing=True,
                      callbacks=[ModelCheckpoint(
                                 dirpath=os.path.join(opts.checkpoints_root, opts.log_name),
                                 filename='model',
                                 every_n_epochs=opts.validation_freq,
                                 save_top_k=1,
                                 save_last=True if opts.train_state == 'baseline' else False,
                                 monitor="iou/cluster",
                                 mode="max")],
                                 **gpu_args)
    trainer.fit(model, train_loader, val_loader)



if __name__ == '__main__':
    args = train_parser()
    print(args)
    main(args)
