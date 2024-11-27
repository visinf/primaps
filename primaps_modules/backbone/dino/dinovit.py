import torch
import torch.nn as nn
# import primaps_modules.backbone.dino.vision_transformer as vits


# class DinoFeaturizer(nn.Module):

#     def __init__(self, arch, patch_size, totrain):
#         super().__init__()
#         self.patch_size = patch_size
#         self.feat_type = "feat"

#         self.model = vits.__dict__[arch](
#             patch_size=patch_size,
#             num_classes=0)
#         for p in self.model.parameters():
#             p.requires_grad = False
#         self.model.eval() #.cuda()
#         if totrain:
#             for p in self.model.parameters():
#                 p.requires_grad = True
#             self.model.train()
#         self.dropout = torch.nn.Dropout2d(p=.1)

#         if arch == "vit_small" and patch_size == 16:
#             url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
#         elif arch == "vit_small" and patch_size == 8:
#             url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
#         elif arch == "vit_base" and patch_size == 16:
#             url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
#         elif arch == "vit_base" and patch_size == 8:
#             url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
#         else:
#             raise ValueError("Unknown arch and patch size")

#         # if pretrained_weights is not None:
#         #     state_dict = torch.load(cfg.pretrained_weights, map_location="cpu")
#         #     state_dict = state_dict["teacher"]
#         #     # remove `module.` prefix
#         #     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#         #     # remove `backbone.` prefix induced by multicrop wrapper
#         #     state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

#         #     # state_dict = {k.replace("projection_head", "mlp"): v for k, v in state_dict.items()}
#         #     # state_dict = {k.replace("prototypes", "last_layer"): v for k, v in state_dict.items()}

#         #     msg = self.model.load_state_dict(state_dict, strict=False)
#         #     print('Pretrained weights found at {} and loaded with msg: {}'.format(cfg.pretrained_weights, msg))
#         # else:
#         print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
#         state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
#         self.model.load_state_dict(state_dict, strict=True)

#     #     if arch == "vit_small":
#     #         self.n_feats = 384
#     #     else:
#     #         self.n_feats = 768
#     #     self.cluster1 = self.make_clusterer(self.n_feats)
#     #     self.proj_type = cfg.projection_type
#     #     if self.proj_type == "nonlinear":
#     #         self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

#     # def make_clusterer(self, in_channels):
#     #     return torch.nn.Sequential(
#     #         torch.nn.Conv2d(in_channels, self.dim, (1, 1)))  # ,

#     # def make_nonlinear_clusterer(self, in_channels):
#     #     return torch.nn.Sequential(
#     #         torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
#     #         torch.nn.ReLU(),
#     #         torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

#     def forward(self, img, n=1, return_class_feat=False):
#         # self.model.eval()
#         with torch.no_grad():
#             assert (img.shape[2] % self.patch_size == 0)
#             assert (img.shape[3] % self.patch_size == 0)

#             # get selected layer activations
#             feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
#             if n == 1:
#                 feat, attn, qkv = feat[0], attn[0], qkv[0]
#             else:
#                 feat, attn, qkv = feat[-n], attn[-n], qkv[-n]



#             feat_h = img.shape[2] // self.patch_size
#             feat_w = img.shape[3] // self.patch_size

#             if self.feat_type == "feat":
#                 image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
#             elif self.feat_type == "KK":
#                 image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
#                 B, H, I, J, D = image_k.shape
#                 image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
#             else:
#                 raise ValueError("Unknown feat type:{}".format(self.feat_type))

#             if return_class_feat:
#                 return image_feat, feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)
#             else:
#                 return image_feat   
        
#             # if self.proj_type is not None:
#             #     code = self.cluster1(self.dropout(image_feat))
#             #     if self.proj_type == "nonlinear":
#             #         code += self.cluster2(self.dropout(image_feat))
#             # else:
#             #     code = image_feat

#             # if self.cfg.dropout:
#             #     return self.dropout(image_feat), code
#             # else:
#             #   return image_feat, code

class DinoFeaturizerv2(nn.Module):

    def __init__(self, arch, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.arch = arch
        if 'v2' in arch:
            self.model = torch.hub.load('facebookresearch/dinov2', arch+str(patch_size))
        elif 'resnet' in arch:
            rn_dino = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            from torchvision.models.feature_extraction import create_feature_extractor    
            return_nodes = {'layer4.2.relu_2': 'out'}
            self.model = create_feature_extractor(rn_dino, return_nodes=return_nodes)
        else:
            self.model = torch.hub.load('facebookresearch/dino:main', arch+str(patch_size))
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()


    def forward(self, img, n=1):
        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            if 'v2' in self.arch:
                image_feat = self.model.get_intermediate_layers(img, n, reshape=True)[n-1]
            elif 'resnet' in self.arch:
                image_feat = self.model(img)['out']
            else:
                image_feat = self.model.get_intermediate_layers(img, n)[-n][:, 1:, :].transpose(1, 2).contiguous()
                image_feat = image_feat.view(image_feat.size(0), image_feat.size(1), img.size(-1)//self.patch_size, img.size(-1)//self.patch_size)

        return image_feat   
        