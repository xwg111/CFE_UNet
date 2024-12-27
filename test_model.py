"""
Our experimental codes are based on 
https://github.com/McGregorWwww/UCTransNet
We thankfully acknowledge the contributions of the authors
"""

import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings
import pickle

warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os

from nets.CFE_UNet import CFE_UNet

import json
from utils import *
import cv2


def show_image_with_dice(predict_save, labs, save_path):

    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))


    return dice_pred, iou_pred
    

def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()

    output = model(input_img.cuda())
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_predict'+model_type+'.jpg')
    input_img.to('cpu')


    input_img = input_img[0].transpose(0,-1).cpu().detach().numpy()
    labs = labs[0]
    output = output[0,0,:,:].cpu().detach().numpy()

    if(False):
        pickle.dump({
            'input':input_img,
            'output':(output>=0.5)*1.0,            
            'ground_truth':labs,
            'dice':dice_pred_tmp,
            'iou':iou_tmp
        },
        open(vis_save_path+'.p','wb'))

    if(True):
        
        plt.figure(figsize=(10,3.3))
        plt.subplot(1,3,1)
        plt.imshow(input_img)
        plt.subplot(1,3,2)
        plt.imshow(labs,cmap='gray')
        plt.subplot(1,3,3)
        plt.imshow((output>=0.5)*1.0,cmap='gray')    
        plt.suptitle(f'Dice score : {np.round(dice_pred_tmp,3)}\nIoU : {np.round(iou_tmp,3)}')
        plt.tight_layout()
        plt.savefig(vis_save_path)
        plt.close()


    return dice_pred_tmp, iou_tmp



if __name__ == '__main__':
    

    test_session = config.test_session
    if config.task_name =="Glas":
        test_num = 80
        model_type = config.model_name
        model_path = "./Glas/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="BUSI":
        test_num = 120
        model_type = config.model_name
        model_path = "./BUSI/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="Kvasir-SEG":
        test_num = 200
        model_type = config.model_name
        model_path = "./Kvasir-SEG/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    
    elif config.task_name =="ISIC2018":
        test_num = 1000
        model_type = config.model_name
        model_path = "./ISIC2018/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="CVC-ClinicDB":
        test_num = 122
        model_type = config.model_name
        model_path = "./CVC-ClinicDB/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    save_path  = config.task_name +'/'+ config.model_name +'/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    vis_path = save_path + 'visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

    fp = open(save_path+'test.result','a')
    fp.write(str(datetime.now())+'\n')

    
    if model_type == 'CFE_UNet':
        config_vit = config.get_CTranS_config()   
        model = CFE_UNet(n_channels=config.n_channels,n_classes=config.n_labels,n_filts=config.n_filts)
    
    else: raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset, tf_test,image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0

    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr=test_data.numpy()
            arr = arr.astype(np.float32())
            lab=test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            height, width = config.img_size, config.img_size
            input_img = torch.from_numpy(arr)
            dice_pred_t,iou_pred_t = vis_and_save_heatmap(model, input_img, None, lab,
                                                          vis_path+str(i)+'.png',
                                               dice_pred=dice_pred, dice_ens=dice_ens)
            dice_pred+=dice_pred_t
            iou_pred+=iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()
    print ("dice_pred",dice_pred/test_num)
    print ("iou_pred",iou_pred/test_num)
    
    fp.write(f"dice_pred : {dice_pred/test_num}\n")
    fp.write(f"iou_pred : {iou_pred/test_num}\n")
    fp.close()



