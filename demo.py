from argparse import ArgumentParser
from typing import Dict
import torch
from PIL import Image
import primaps_modules.transforms as transforms
from primaps_modules.primaps import PriMaPs
from primaps_modules.backbone.dino.dinovit import DinoFeaturizerv2
from primaps_modules.visualization import visualize_demo
# set seeds
torch.manual_seed(0)
torch.cuda.manual_seed(0)
    
    
    
def main(opts: Dict):
    '''
    Demo to visualize PriMaPs for a single image.
    '''    
    # get SLL image encoder and primaps module
    net = DinoFeaturizerv2(opts.backbone_arch, opts.backbone_patch)
    net.to(opts.device)
    primaps_module = PriMaPs(threshold=opts.threshold,
                             ignore_id=255)
    
    # get transforms
    demo_transforms = transforms.Compose([transforms.ToTensor(),
                    transforms.Resize(opts.validation_resize), 
                    transforms.CenterCrop([opts.validation_resize[0], opts.validation_resize[0]]),
                    transforms.Normalize()])
    

    # load image and apply transforms
    img = Image.open(opts.image_path)
    img, _ = demo_transforms(img, torch.zeros(img.size))
    img.to(opts.device)
    # get SSL features
    feats = net(img.unsqueeze(0).to(opts.device), n=1).squeeze()
    # get primaps pseudo labels
    primaps = primaps_module._get_pseudo(img, feats, torch.zeros(img.shape[1:]))
    # visualize overlay
    Image.fromarray(visualize_demo(img, primaps)).save('demo.png')
    print('Image saved as demo.png')
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--backbone-arch", 
                        type=str, 
                        default=['dino_vits', 'dino_vitb', 'dinov2_vits', 'dinov2_vitb'][1],
                        help='backbone architecture')
    parser.add_argument("--backbone-patch", 
                        type=int, 
                        default=[8, 14, 16][0],
                        help='patch size of the vit backbone')
    parser.add_argument("--validation-resize", 
                        nargs='+', 
                        type=int, 
                        default=[[320], [322]][0],
                        help='resize images to this size')
    parser.add_argument("--threshold", 
                        type=float, 
                        default=0.35,
                        help='primaps threshold')
    parser.add_argument("--device", 
                        type=str, 
                        default='cuda:0',
                        help='device to use')
    parser.add_argument("--image-path",
                        type=str,
                        default=['assets/demo_examples/IMG_0709.jpg', 'assets/demo_examples/cityscapes_example.png', 'assets/demo_examples/coco_example.jpg', 'assets/demo_examples/potsdam_example.png'][0],
                        help='path to images')
    args = parser.parse_args()
    print(args)
    main(args)
