import argparse
import os


def check_parser(args):
    if os.path.basename(args.dataset_root) == 'cityscapes':
        args.num_classes = 27
        dataname = 'CS'
    elif os.path.basename(args.dataset_root) == 'cocostuff':
        args.num_classes = 27
        dataname = 'COCO'
    elif os.path.basename(args.dataset_root) == 'potsdam':
        args.num_classes = 3
        dataname = 'PD'
    else:
        raise NotImplementedError
    if 'vits' in args.backbone_arch:
        args.backbone_dim = 384
    elif 'vitb' in args.backbone_arch:
        args.backbone_dim = 768
    else:
        raise NotImplementedError
    
    args.log_name = dataname+'_'+str(args.backbone_arch)+str(args.backbone_patch)+'_'+args.log_name
    
    if args.cluster_ckpt_path != '':
        if args.cluster_ckpt_path[-4:] != 'ckpt':
            args.cluster_ckpt_path = os.path.join(args.cluster_ckpt_path, dataname+'_'+str(args.backbone_arch)+str(args.backbone_patch)+'_init', 'last.ckpt')
        
    return args
    

# parser train.py
def base_parser():
    parser = argparse.ArgumentParser()
    ### Dataset and Backbone 
    parser.add_argument("--dataset-root", type=str, default=['/fastdata/ohahn/datasets/cityscapes', '/fastdata/ohahn/datasets/cocostuff', '/fastdata/ohahn/datasets/potsdam'][0])
    parser.add_argument("--num-classes", type=int, default=[27, 3][0])
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--checkpoints-root", type=str, default='/visinf/projects/ohahn/checkpoints/') 
    parser.add_argument("--logg-root", type=str, default='/visinf/projects/ohahn/checkpoints/')
    parser.add_argument("--backbone-dim", type=int, default=[384, 768][0])
    parser.add_argument("--backbone-arch", type=str, default=['dino_vits', 'dino_vitb', 'dinov2_vits', 'dinov2_vitb'][1])
    parser.add_argument("--backbone-patch", type=int, default=[8, 14, 16][0])
    parser.add_argument("--dino-block", type=int, default=1, help='block outputing the used feature') 
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--validation-resize", nargs='+', type=int, default=[[320], [322]][0])
    parser.add_argument("--crop-size", nargs='+', type=int, default=[[224], [320], [322]][1])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--num-steps", type=int, default=7000)
    parser.add_argument("--cluster-ckpt-path", type=str, default='') 
    parser.add_argument('--stop-criterion', type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)   
    parser.add_argument("--gpu-ids", nargs='+', type=int, default=([0])) 
    return parser


def train_parser():
    parser = base_parser()
    ### Augmentations 
    parser.add_argument("--student-augs", action='store_true', default=False)
    parser.add_argument("--augs-photometric", action='store_true', default=False)
    parser.add_argument("--augs-randcrop-scale", nargs='+', type=float, default=[[0.25, 1.], [0.8, 1.0], [1., 1.]][-1])
    parser.add_argument("--augs-randcrop-ratio", nargs='+', type=float, default=[[4/5, 5/4], [1., 1.]][-1])
    ### Linear Probing 
    parser.add_argument("--linear-lr", type=float, default=5e-3)
    ### Mask Proposals 
    parser.add_argument("--train-state", type=str, default=['baseline', 'method'][-1])
    parser.add_argument('--pca-iter', type=int, default=100)
    parser.add_argument('--pca-q', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--gain', type=float, default=10.0)
    # Pre-computed masks
    parser.add_argument("--precomp-primaps", action='store_true', default=False)
    parser.add_argument("--precomp-primaps-root", type=str, default='')
    ### Cluster Head 
    parser.add_argument("--ema-update-step", type=int, default=-1)
    parser.add_argument("--ema-decay", type=float, default=0.98)
    parser.add_argument("--cluster-lr", type=float, default=5e-3)
    parser.add_argument("--num-batch-pcainit", type=int, default=92)
    parser.add_argument("--cluster-train", action='store_true', default=False) 
    parser.add_argument("--stego-ckpt", type=str, default='')    
    parser.add_argument("--hp-ckpt", type=str, default='')
    parser.add_argument("--hp-opt", type=str, default='/visinf/home/ohahn/code/HP/json/server/cocostuff_eval.json')
    ### Segmentation Head 
    parser.add_argument("--seghead-lr", type=float, default=5e-3)
    parser.add_argument("--seghead-arch", type=str, default=['linear', 'mlp'][0])
    parser.add_argument("--seghead-focalloss", type=float, default=2.0)
    ### PCA Init 
    parser.add_argument("--pcainit", action='store_true', default=False)
    ### Training misc      
    parser.add_argument("--log-name", type=str, default='Experiment')
    parser.add_argument("--validation-freq", type= int, default=1)
    args = check_parser(parser.parse_args())
    return args


# parser test.py
def test_parser():
    parser = base_parser()
    parser.add_argument("--checkpoint-path", type=str, default='')
    return parser.parse_args()
