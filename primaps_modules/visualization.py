import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from cityscapesscripts.helpers.labels import labels as cs_labels
from datasets.cityscapes import get_cs_labeldata
from datasets.cocostuff import get_coco_labeldata
from datasets.potsdam import get_pd_labeldata

sys.path.append(os.getcwd())
import primaps_modules.transforms as transforms


def visualize_segmentation(img = None, 
                           label = None, 
                           linear = None, 
                           mlp = None, 
                           cluster = None, 
                           dataset_name = None,
                           additional = None, 
                           additional_name = None, 
                           additional2 = None, 
                           additional_name2 = None, 
                           legend = None,
                           name = None):


    if dataset_name == "cityscapes":
        colormap = np.array([
            [128, 64, 128],
            [244, 35, 232],
            [250, 170, 160],
            [230, 150, 140],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [180, 165, 180],
            [150, 100, 100],
            [150, 120, 90],
            [153, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 0, 90],
            [0, 0, 110],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
            [0, 0, 0],
            [220, 220, 220]])
    elif dataset_name == "cocostuff":
        colormap = get_coco_labeldata()[-1]


    orig_h, orig_w = label.cpu().shape[-2:]
    img = img.cpu().squeeze(0).numpy().transpose(1, 2, 0)
    img = (img-img.min())/(img-img.min()).max()
    label = label.cpu().squeeze(0).numpy().transpose(1, 2, 0)
    #transforms.labelIdsToTrainIds(source="cityscapes", target="cityscapes")

    label[label == 255] = 27
    colored_label = colormap[label.flatten()]
    colored_label = colored_label.reshape(orig_h, orig_w, 3)

    num_subplots = 3
    if linear != None: num_subplots += 1
    if mlp != None: num_subplots += 1 
    if additional != None: num_subplots += 1 
    if additional2 != None: num_subplots += 1         


    fig = plt.figure(figsize=(8, 2), dpi=200)
    fig.tight_layout()
    plt.axis('off')
    plt.subplot(1, num_subplots, 1)
    plt.gca().set_title('Image')
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1, num_subplots, 2)
    plt.gca().set_title('Ground Truth')
    plt.imshow(colored_label)
    plt.axis("off")
    i = 3
    if linear != None:
        linear = linear.cpu().numpy().transpose(1, 2, 0).astype('uint8')
        linear = colormap[linear.flatten()].reshape(linear.shape[0], linear.shape[1], 3)
        plt.axis("off")
        plt.subplot(1, num_subplots, i)
        plt.gca().set_title('Linear')
        plt.imshow(linear)
        i+=1

    if mlp != None:
        mlp = mlp.cpu().numpy().transpose(1, 2, 0).astype('uint8')    
        mlp = colormap[mlp.flatten()].reshape(mlp.shape[0], mlp.shape[1], 3)
        plt.axis("off")
        plt.subplot(1, num_subplots, i)
        plt.gca().set_title('MLP')
        plt.imshow(mlp)
        plt.axis("off")
        i+=1

    if cluster != None:
        cluster = cluster.cpu().numpy().transpose(1, 2, 0).astype('uint8')
        cluster = colormap[cluster.flatten()].reshape(cluster.shape[0], cluster.shape[1], 3)
        plt.axis("off")
        plt.subplot(1, num_subplots, i)
        plt.gca().set_title('Cluster')
        plt.imshow(cluster)
        plt.axis("off")
        i+=1

    if additional != None:
        #additional = additional.cpu().numpy()
        additional = additional.cpu().numpy().transpose(1, 2, 0).astype('uint8')
        additional = colormap[additional.flatten()].reshape(additional.shape[0], additional.shape[1], 3)
        plt.axis("off")
        plt.subplot(1, num_subplots, i)
        plt.gca().set_title(additional_name)
        plt.imshow(additional)
        plt.axis("off")
        i+=1

    if additional2 != None:
        additional2 = additional2.cpu().numpy()
        plt.axis("off")
        plt.subplot(1, num_subplots, i)
        plt.gca().set_title(additional_name2)
        plt.imshow(additional2)
        plt.axis("off")
        i+=1


    # if legend != None:  
    #     from matplotlib.lines import Line2D

    #     legend_elements = [Line2D([0], [0], color=np.array(cls[7])/255, lw=4, label=cls[0]) for cls in cs_labels[7:-1]]

    #     # Create the figure
    #     #fig, ax = plt.subplots()
    #     plt.legend(handles=legend_elements, loc='right')



    if name != None: plt.savefig(name)
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close('all')

    return data



def visualize_confusion_matrix(cls_names, meter, name=None):
    # plot of confusion matrix
    conf_matrix = (meter.histogram/meter.histogram.sum(dim=0))
    conf_matrix = np.array(conf_matrix.cpu(), dtype=np.float16)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.matshow(torch.Tensor(conf_matrix).fill_diagonal_(0), cmap=plt.cm.Blues, alpha=0.8)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=(conf_matrix[i, j]*100).round(1), va='center', ha='center', size='large')
    ax.set_xticks(list(range(cls_names.__len__())))
    ax.set_xticklabels(cls_names, rotation=90, ha='center', fontsize=12)
    ax.set_yticks(list(range(cls_names.__len__())))
    ax.set_yticklabels(cls_names, fontsize=12)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    
    if name != None: plt.savefig(name)
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close('all')
    return data





def batch_visualize_segmentation(img = None, 
                                 label = None, 
                                 in1 = None,  
                                 in2 = None, 
                                 in3 = None,
                                 in4 = None, 
                                 dataset_name = None):


    if dataset_name == "cityscapes":
        colormap = get_cs_labeldata()[-1]
    elif dataset_name == "cocostuff":
        colormap = get_coco_labeldata()[-1]
    elif dataset_name == "potsdam":
        colormap = get_pd_labeldata()[-1]

    def _vis_one_img(idx, img, label, ins):

        orig_h, orig_w = label.cpu().shape[-2:]
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = (img-img.min())/(img-img.min()).max()
        label = label.cpu().numpy().transpose(1, 2, 0)
        label[label > 27] = 27
        colored_label = colormap[label.flatten()].reshape(orig_h, orig_w, 3)

        num_subplots = sum([1 for x in [in1, in2, in3, in4] if x != None]) + 2      

        fig = plt.figure(figsize=(10, 2), dpi=150)
        fig.tight_layout()
        plt.axis('off')
        plt.subplot(1, num_subplots, 1)
        if idx == 0: plt.gca().set_title('Image')
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, num_subplots, 2)
        if idx == 0: plt.gca().set_title('Ground Truth')
        plt.imshow(colored_label)
        plt.axis("off")
        if ins != None:
            i = 3
            for input in ins:
                vis = input[1].cpu().numpy().transpose(1, 2, 0).astype('uint8')
                vis = colormap[vis.flatten()].reshape(vis.shape[0], vis.shape[1], 3)
                plt.axis("off")
                plt.subplot(1, num_subplots, i)
                if idx == 0: plt.gca().set_title(input[0])
                plt.imshow(vis)
                plt.axis("off")
                i+=1

        fig.canvas.draw()
        plt.close('all')
        one_vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        one_vis = one_vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close('all')
        return one_vis
    
    imgs = []
    for idx, (data) in enumerate(zip(img, label)):
        imgs.append(_vis_one_img(idx, data[0], data[1], [[i[0], i[1][idx].unsqueeze(0)] for i in [in1, in2, in3, in4] if i!=None]))

    return np.vstack(imgs)  



def visualize_single_masks(img, 
                           label, 
                           data,
                           dataset_name = None):


    if dataset_name == "cityscapes":
        colormap = get_cs_labeldata()[-1]
    elif dataset_name == "cocostuff":
        colormap = get_coco_labeldata()[-1]
    elif dataset_name == "potsdam":
        colormap = get_pd_labeldata()[-1]
        
        
    fig = plt.figure(figsize=(data['sim'].__len__()*2, 7*2), dpi=150)
    fig.tight_layout()
    for indx, (sim, nnsim, nnsim_thresh, crf, pamr, mask) in enumerate(zip(data['sim'], data['nnsim'], data['nnsim_tresh'], data['crf'], data['pamr'], data['outmask'])):                    
        rows = data['sim'].__len__()
        cols = 8
        plotlabel=colormap[label.squeeze(0).squeeze(0).int().cpu()]
        plt.subplot(rows, cols, 1+(indx*cols))
        img = (img-img.min())/(img.max()-img.min())
        if indx == 0: plt.title('Image')
        plt.imshow(img.squeeze(0).permute(1, 2, 0).cpu())
        plt.axis('off')
        plt.subplot(rows, cols, 2+(indx*cols))
        if indx == 0: plt.title('GT')
        plt.imshow(plotlabel)
        plt.axis('off')
        plt.subplot(rows, cols, 3+(indx*cols))
        if indx == 0: plt.title('1.Eig')
        plt.imshow(sim.cpu().numpy())
        plt.axis('off')
        plt.subplot(rows, cols, 4+(indx*cols))
        if indx == 0: plt.title('1.EigNN')
        plt.imshow(nnsim.cpu().numpy())
        plt.axis('off')
        plt.subplot(rows, cols, 5+(indx*cols))
        if indx == 0: plt.title('+Thresh')
        plt.imshow(nnsim_thresh)
        plt.axis('off')
        plt.subplot(rows, cols, 6+(indx*cols))
        if indx == 0: plt.title('+CRF')
        plt.imshow(crf)
        plt.axis('off')
        plt.subplot(rows, cols, 7+(indx*cols))
        if indx == 0: plt.title('PAMR')
        plt.imshow(pamr.squeeze().cpu().numpy())
        plt.axis('off')
        plt.subplot(rows, cols, 8+(indx*cols))
        if indx == 0: plt.title('Mask')
        mask[0, 0] = 0
        plt.imshow(mask.numpy(), cmap='Greys')
        plt.axis('off')
        # plt.savefig(str(idx)+'.png', tight_layout=True)
        
    
    fig.canvas.draw()
    plt.close('all')
    one_vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    one_vis = one_vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close('all')
    return one_vis





def visualize_pseudo_paper(img, 
                           label, 
                           pseudo_gt,
                           pseudo_plain,
                           dataset_name = None,
                           save_name = None):


    if dataset_name == "cityscapes":
        colormap = get_cs_labeldata()[-1]
    elif dataset_name == "cocostuff":
        colormap = get_coco_labeldata()[-1]
    elif dataset_name == "potsdam":
        colormap = get_pd_labeldata()[-1]       
        
    
    np.random.seed(0)
    cb_colomap = np.array([list(np.random.randint(0, 255, size=(1,3))[0]) for _ in range(400)]+[[0, 0, 0]])
    pseudo_plain = pseudo_plain.int().cpu()
    pseudo_plain[pseudo_plain==255] = 400
    pseudo_plain = cb_colomap[pseudo_plain.int().cpu()].squeeze()
        

        
        
    fig = plt.figure(figsize=(8, 2), dpi=150)
    fig.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.5,
                        top=0.5,
                        wspace=0.05,
                        hspace=0.0)

    plt.subplot(1, 4, 1)
    img = (img-img.min())/(img.max()-img.min())
    img = img.squeeze(0).permute(1, 2, 0).cpu()
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plotlabel=colormap[label.squeeze(0).squeeze(0).int().cpu()]
    plt.imshow(plotlabel)
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plotpseudo=colormap[pseudo_gt.squeeze(0).squeeze(0).int().cpu()]
    # pseudo_plain = np.array(pseudo_plain.cpu(), dtype=np.int16).squeeze()
    # plotpseudo = mark_boundaries(plotlabel/255, pseudo_plain, color=(1, 1, 1))
    plt.imshow(plotpseudo)
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(pseudo_plain)
    plt.axis('off')
    plt.savefig(save_name+'.pdf', bbox_inches='tight', pad_inches=0.0)
    

    save_name_single = os.path.join(os.path.dirname(save_name), 'singleimgs/')
    os.makedirs(os.path.dirname(save_name_single), exist_ok=True)
    for i, n in zip([img, plotlabel, plotpseudo, pseudo_plain], ['img', 'gt', 'pseudo', 'pseudoc']):
        fig = plt.figure(figsize=(2, 2), dpi=300)
        plt.imshow(i)
        plt.axis('off')
        plt.savefig(os.path.join(save_name_single, os.path.split(save_name)[-1]+'_'+n+'.png'), bbox_inches='tight', pad_inches=0.0)
        
        
        
        
        
def logits_to_image(logits = None, 
                    img = None,
                    label = None,
                    dataset_name = None,
                    save_path = None,
                    save_imggt = False):


    if dataset_name == "cityscapes":
        colormap = get_cs_labeldata()[-1]
    elif dataset_name == "cocostuff":
        colormap = get_coco_labeldata()[-1]
    elif dataset_name == "potsdam":
        colormap = get_pd_labeldata()[-1]
        
    vis = logits.cpu().numpy().transpose(1, 2, 0).astype('uint8')
    vis = colormap[vis.flatten()].reshape(vis.shape[0], vis.shape[1], 3)
        
    fig = plt.figure(figsize=(2, 2), dpi=400)
    fig.tight_layout()
    plt.subplot(1, 1, 1)
    plt.imshow(vis)
    plt.axis("off")
    plt.savefig(save_path+'_pred.png', bbox_inches='tight', pad_inches=0.0)
    plt.close('all')
    
    if save_imggt:
        orig_h, orig_w = label.cpu().shape[-2:]
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = (img-img.min())/(img-img.min()).max()
        label = label.cpu().numpy().transpose(1, 2, 0)
        label[label > 27] = 27
        colored_label = colormap[label.flatten()].reshape(orig_h, orig_w, 3)

        fig = plt.figure(figsize=(2, 2), dpi=400)
        fig.tight_layout()
        plt.subplot(1, 1, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.savefig(save_path+'_img.png', bbox_inches='tight', pad_inches=0.0)
        plt.close('all')
        
        fig = plt.figure(figsize=(2, 2), dpi=400)
        fig.tight_layout()
        plt.subplot(1, 1, 1)
        plt.imshow(colored_label)
        plt.axis("off")
        plt.savefig(save_path+'_gt.png', bbox_inches='tight', pad_inches=0.0)
        plt.close('all')
        
        
    
    
class Vis_Demo():
    def __init__(self):
        super(Vis_Demo, self).__init__()
        self.colormap = get_coco_labeldata()[-1]

    def apply_colors(self, logits):
        vis = logits.cpu().numpy().transpose(1, 2, 0).astype('uint8')
        vis = self.colormap[vis.flatten()].reshape(vis.shape[0], vis.shape[1], 3)
        return vis
    
    
    
def visualize_demo(img, pseudo, alpha = 0.5):
    np.random.seed(0)
    cb_colomap = np.array([list(np.random.randint(0, 255, size=(1,3))[0]) for _ in range(400)]+[[0, 0, 0]])
    pseudo_plain = pseudo.long().cpu().numpy()
    pseudo_plain[pseudo_plain==255] = 400
    pseudo_plain = cb_colomap[pseudo_plain].squeeze()
    
    img = transforms.UnNormalize()(img)*255
    img = img.permute(1, 2, 0).long().cpu().numpy()
    out = alpha*img + (1-alpha)*pseudo_plain
    
    return np.array(out, dtype=np.uint8)
        
