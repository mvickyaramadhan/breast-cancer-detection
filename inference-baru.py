#codin:utf8
# - *- coding: utf- 8 - *-
from config import opt
import os
import models
#import face_alignment
#from skimage import io
from torch.autograd import Variable
from torchnet import meter
from utils import Visualizer
from tqdm import tqdm
from torchvision import transforms
import torchvision
import torch
from torchsummary import summary
import json
import numpy as np
import cv2
import csv
from efficientnet_pytorch import EfficientNet # Test EfficientNet 10162020.

class DataHandle():

    def __init__(self,scale=2.7,image_size=224,use_gpu=False,transform=None,data_source = None):
        self.transform = transform
        self.scale = scale
        self.image_size = image_size
        #use CPU for face alignment, since the GPU memory is insufficient. Kahlil 03302020.
        #self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,device='cpu')
        # if use_gpu:
        #     self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        # else:
        #     self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,device='cpu')

    def thresholded(self, center, pixels):
        out = []
        for a in pixels:
            if a >= center:
                out.append(1)
            else:
                out.append(0)
        return out

    def get_pixel_else_0(self, l, idx, idy, default=0):
        try:
            return l[idx,idy]
        except IndexError:
            return default

    def det_img(self,imgdir):
        input = io.imread(imgdir)
        preds = self.fa.get_landmarks(input)
        if 0:
            for pred in preds:
                img = cv2.imread(imgdir)
                print('ldmk num:', pred.shape[0])
                for i in range(pred.shape[0]):
                    x,y = pred[i]
                    print(x,y)
                    cv2.circle(img,(x,y),1,(0,0,255),-1)
                cv2.imshow('-',img)
                cv2.waitKey()
        return preds
    def crop_with_ldmk(self,image, landmark):
        ct_x, std_x = landmark[:,0].mean(), landmark[:,0].std()
        ct_y, std_y = landmark[:,1].mean(), landmark[:,1].std()

        std_x, std_y = self.scale * std_x, self.scale * std_y

        src = np.float32([(ct_x, ct_y), (ct_x + std_x, ct_y + std_y), (ct_x + std_x, ct_y)])
        dst = np.float32([((self.image_size -1 )/ 2.0, (self.image_size -1)/ 2.0),
                  ((self.image_size-1), (self.image_size -1 )),
                  ((self.image_size -1 ), (self.image_size - 1)/2.0)])
        retval = cv2.getAffineTransform(src, dst)
        result = cv2.warpAffine(image, retval, (self.image_size, self.image_size), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT)
        return result

    def get_data(self,image_path):#第二步装载数据，返回[img,label]
        img = cv2.imread(image_path)
        print(image_path)
        # ldmk = np.asarray(self.det_img(image_path),dtype=np.float32)
        # if 0:
        #     for pred in ldmk:
        #         for i in range(pred.shape[0]):
        #             x,y = pred[i]
        #             cv2.circle(img,(x,y),1,(0,0,255),-1)
        # ldmk = ldmk[np.argsort(np.std(ldmk[:,:,1],axis=1))[-1]]
        # img =self.crop_with_ldmk(img, ldmk)
        
        # cv2.imwrite('data_ori.jpg',img)
        # img_test = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # cv2.imwrite('data_yuv.jpg',img_test)
        # y1, u1, v1 = cv2.split(img_test)
        # cv2.imwrite('data_channel.jpg',v1)
        # transformed_img = v1.copy()
        # for x in range(0, len(v1)):
        #     for y in range(0, len(v1[0])):
        #         center        = v1[x,y]
        #         top_left      = self.get_pixel_else_0(v1, x-1, y-1)
        #         top_up        = self.get_pixel_else_0(v1, x, y-1)
        #         top_right     = self.get_pixel_else_0(v1, x+1, y-1)
        #         right         = self.get_pixel_else_0(v1, x+1, y )
        #         left          = self.get_pixel_else_0(v1, x-1, y )
        #         bottom_left   = self.get_pixel_else_0(v1, x-1, y+1)
        #         bottom_right  = self.get_pixel_else_0(v1, x+1, y+1)
        #         bottom_down   = self.get_pixel_else_0(v1, x,   y+1 )

        #         values = self.thresholded(center, [top_left, top_up, top_right, right, bottom_right,
        #                                     bottom_down, bottom_left, left])

        #         weights = [1, 2, 4, 8, 16, 32, 64, 128]
        #         res = 0
        #         for a in range(0, len(values)):
        #             res += weights[a] * values[a]

        #         transformed_img.itemset((x,y), res)

        #     #print (x)

        # cv2.imwrite('thresholded_image.jpg', transformed_img)
        # #cv2.imshow('data.jpg',img)
        # #cv2.waitKey()
        # if 0:
        #     cv2.imshow('crop face',img)
        #     cv2.waitKey(0)

        return np.transpose(np.array(img, dtype = np.float32), (2, 0, 1)), image_path

    def __len__(self):
        return len(self.img_label)
    def crop_with_ldmk(self,image, landmark):
        ct_x, std_x = landmark[:,0].mean(), landmark[:,0].std()
        ct_y, std_y = landmark[:,1].mean(), landmark[:,1].std()

        std_x, std_y = self.scale * std_x, self.scale * std_y

        src = np.float32([(ct_x, ct_y), (ct_x + std_x, ct_y + std_y), (ct_x + std_x, ct_y)])
        dst = np.float32([((self.image_size -1 )/ 2.0, (self.image_size -1)/ 2.0),
                  ((self.image_size-1), (self.image_size -1 )),
                  ((self.image_size -1 ), (self.image_size - 1)/2.0)])
        retval = cv2.getAffineTransform(src, dst)
        result = cv2.warpAffine(image, retval, (self.image_size, self.image_size), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT)
        cv2.imwrite('crop.jpg',result)
        return result

output_list = []
header = ['output']

def inference(**kwargs):
    counting = 0
    counter = 0
    import glob
    images = glob.glob(kwargs['images'])
    path = kwargs['images']
    assert len(images)>0
    data_handle = DataHandle(
                        scale = opt.cropscale,
                        use_gpu = opt.use_gpu,
			transform = None,
			data_source='none')
    pths = glob.glob('checkpoints/%s/*.pth'%(opt.model))
    pths.sort(key=os.path.getmtime,reverse=True)
    print(pths)
    opt.parse(kwargs)
    # 模型
    opt.load_model_path=pths[0] #remark 10162020
    
    #model = getattr(models, opt.model)().eval() # remark 10162020
    model = EfficientNet.from_pretrained('efficientnet-b4') # Test EfficientNet 10162020.

    assert os.path.exists(opt.load_model_path)
    if opt.load_model_path:
       #model.load(opt.load_model_path) # remark 10162020
       model = torch.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()
    model.train(False)
    fopen = open('result/inference.txt','w')
    tqbar = tqdm(enumerate(images),desc='Inference with %s'%(opt.model))
    for idx,imgdir in tqbar:
        data,_ = data_handle.get_data(imgdir)
        data = data[np.newaxis,:]
        data = torch.FloatTensor(data)
        with torch.no_grad():
            if opt.use_gpu:
                data =  data.cuda()
            outputs = model(data)
            outputs = torch.softmax(outputs,dim=-1)
            preds = outputs.to('cpu').numpy()
            attack_prob = preds[:,opt.ATTACK]
            tqbar.set_description(desc = 'Inference %s attack_prob=%f with %s'%(imgdir, attack_prob, opt.model))
   #         print('Inference %s attack_prob=%f'%(imgdir, attack_prob),file=fopen)
    #fopen.close()

            output_list.append(attack_prob)
            counting += attack_prob
            counter += 1
    average = counting/counter
    print(counter)
    print(counting)
    if 'positive' in path:
        average = 1 - average
    print(average)





#df = pd.DataFrame(output_list, columns = ['name'])
#df['name'] = output_list
#print(df)

    # buat jumlah rata2 nilai prob
    #a = 0;
 
        #hasil = a + attack_prob;
        #print(hasil/826)

# Inference detlandmark/inferences/negative/9365192L.png attack_prob=0.490185 with MyresNet50: : 821it [00:06, 124.07it/s]detlandmark/inferences/negative/9726175R.png

def help():
    '''
    打印帮助的信息： python file.py help
    '''

    print('''
    usage : python {0} <function> [--args=value,]
    <function> := train | test | help
    example:
           python {0} train --env='env0701' --lr=0.01
           python {0} test --dataset='path/to/dataset/root/'
           python {0} inference --images='image dirs'
           python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__=='__main__':
    import fire
    fire.Fire()
