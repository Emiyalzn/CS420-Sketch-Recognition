import numpy as np
from torch.utils.data import DataLoader
from dataset.dataset import SketchDataset
from utils.seq2png import draw_strokes
import cairosvg
import wx
from datetime import datetime
import os

class HumanInteractor(wx.Frame): 
    categories = np.array(['bear', 'camel', 'cat', 'cow', 'crocodile', 
                  'dog', 'elephant', 'flamingo', 'giraffe', 'hedgehog', 
                  'horse', 'kangaroo', 'lion', 'monkey', 'owl', 
                  'panda', 'penguin', 'pig', 'raccoon', 'rhinoceros', 
                  'sheep', 'squirrel', 'tiger', 'whale', 'zebra'])

    categories_chinese = ['熊', '骆驼', '猫', '牛', '鳄鱼', 
                          '狗', '象', '火烈鸟', '长颈鹿', '刺猬', 
                          '马', '袋鼠', '狮子', '猴子', '猫头鹰', 
                          '熊猫', '企鹅', '猪', '浣熊', '犀牛', 
                          '羊', '松鼠', '老虎', '鲸鱼', '斑马']
    
    def __init__(self, data_seq_dir, log_root_dir='./results', categories=None, parent=None, title="Quick, Draw! Human Recognition Experiment"):
        super(HumanInteractor, self).__init__(parent, title=title, size=(400, 700))

        self.log_dir = os.path.join(log_root_dir, f'human_exp-{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        os.makedirs(self.log_dir)

        if (categories is None):
            self.categories = np.array(HumanInteractor.categories)
        else:
            self.categories = np.array(categories)
        
        self.categories_chinese = []
        self.category_values = dict()
        
        for i, ctg in enumerate(self.categories):
            idx = np.where(ctg == HumanInteractor.categories)[0][0]
            self.categories_chinese.append(HumanInteractor.categories_chinese[idx])
            self.category_values[f'{ctg} ({self.categories_chinese[-1]})'] = i

        self.confusion_matrix = np.zeros((len(self.categories), len(self.categories)), dtype=np.int32)

        self.testSet = SketchDataset(
            mode='test',
            data_seq_dir=data_seq_dir,
            data_img_dir=None,
            categories=self.categories,
            disable_augmentation=True
        )
        self.testLoader = DataLoader(self.testSet, batch_size=1, shuffle=True, pin_memory=False)
        self.it = iter(self.testLoader)

        self.img = None
        self.txt = None
        self.cnt = 0
        
        self.next()
        self.initUI()
    
    def next(self):
        cur = next(self.it)
        data = cur[0][0].numpy().astype(np.int32, 'C')
        self.curCategory = cur[-1].item()

        draw_strokes(data, self.log_dir+'/tmp.svg', width=224)
        with open(self.log_dir+'/tmp.svg', 'r') as f_in:
            svg = f_in.read()
            f_in.close()
        cairosvg.svg2png(bytestring=svg, write_to=self.log_dir+'/tmp.png')
        
        self.image_data = wx.Image(self.log_dir+'/tmp.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        
        if (self.img is not None):
            self.img.SetBitmap(self.image_data)
        if (self.txt is not None):
            self.txt.SetLabelText(f'Already recognized {self.cnt} images.')
        self.cnt += 1
        
        print(f'Current: {self.categories_chinese[self.curCategory]} ({self.categories[self.curCategory]})')
    
    def initUI(self):
        panel = wx.Panel(self)
        
        self.rbox = wx.RadioBox(panel, label="选项 (Options)", choices=list(self.category_values.keys()), style=wx.RA_SPECIFY_ROWS)
        # self.rbox.Bind(wx.EVT_RADIOBOX, self.onRadioBoxChange)
        
        self.btn = wx.Button(panel, -1, "确认 (Confirm)", pos=(225, 100))
        self.btn.Bind(wx.EVT_BUTTON, self.onClicked)
        
        self.img = wx.StaticBitmap(panel, bitmap=self.image_data, pos=(160, 200))
        
        self.txt = wx.StaticText(panel, label=f'Already recognized 0 images.', pos=(175, 500))
        
        self.Center()
        self.Show(True)

    # def onRadioBoxChange(self, e):
    #     stringChosen = self.rbox.GetStringSelection()
    #     categoryChosen = HumanInteractor.categoryValues[stringChosen]
    #     print(stringChosen, categoryChosen)
    
    def onClicked(self, e):
        stringChosen = self.rbox.GetStringSelection()
        categoryChosen = self.category_values[stringChosen]
        print(f'Chosen: {stringChosen} {categoryChosen}')
        
        self.confusion_matrix[self.curCategory, categoryChosen] += 1
        # confusion_matrix[i, j]: True label is `i`, Chosen label is `j`
        self.next()
    
    def __del__(self):
        print(self.confusion_matrix)
        
        for k, i in self.category_values.items():
            TP = self.confusion_matrix[i, i]
            TN = np.sum(self.confusion_matrix[:i, :i]) + np.sum(self.confusion_matrix[:i, i+1:]) \
               + np.sum(self.confusion_matrix[i+1:, :i]) + np.sum(self.confusion_matrix[i+1:, i+1:])
    
            T = np.sum(self.confusion_matrix[i])    # = TP + FN
            P = np.sum(self.confusion_matrix[:, i]) # = TP + FP
            
            FN = T - TP
            FP = P - TP
            
            accuracy = (TP + TN) / (TP + FN + FP + TN)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)
            
            print(f'{k}: Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1: {f1}')
        
        np.save(self.log_dir+'/result.npy', self.confusion_matrix)


if __name__ == '__main__':
    ex = wx.App()
    HumanInteractor(data_seq_dir='/home/purewhite/workspace/cs420/project/data/dataset_raw',
                    log_root_dir='/home/purewhite/workspace/cs420/project/CS420-Proj/results',
                    categories=[
                        'bear', 'camel', 'cat', 'cow', 'crocodile', 
                        # 'dog', 'elephant', 'flamingo', 'giraffe', 'hedgehog', 
                        # 'horse', 'kangaroo', 'lion', 'monkey', 'owl', 
                        # 'panda', 'penguin', 'pig', 'raccoon', 'rhinoceros', 
                        # 'sheep', 'squirrel', 'tiger', 'whale', 'zebra'
                  ])
    ex.MainLoop()