import numpy as np
from torch.utils.data import DataLoader
from dataset.dataset import SketchDataset
from utils.seq2png import draw_strokes
import cairosvg
import wx

class HumanInteractor(wx.Frame):
    categoryValues = {
        '熊 (bear)': 0,
        '骆驼 (camel)': 1,
        '猫 (cat)': 2,
        '牛 (cow)': 3,
        '鳄鱼 (crocodile)': 4,
        '狗 (dog)': 5,
        '象 (elephant)': 6,
        '火烈鸟 (flamingo)': 7,
        '长颈鹿 (giraffe)': 8,
        '刺猬 (hedgehog)': 9,
        '马 (horse)': 10,
        '袋鼠 (kangaroo)': 11,
        '狮子 (lion)': 12,
        '猴子 (monkey)': 13,
        '猫头鹰 (owl)': 14,
        '熊猫 (panda)': 15,
        '企鹅 (penguin)': 16,
        '猪 (pig)': 17,
        '浣熊 (raccoon)': 18,
        '犀牛 (rhinoceros)': 19,
        '羊 (sheep)': 20,
        '松鼠 (squirrel)': 21,
        '老虎 (tiger)': 22,
        '鲸鱼 (whale)': 23,
        '斑马 (zebra)': 24,
    }
    
    def __init__(self, parent, title="Quick, Draw! Human Recognition Experiment"):
        super(HumanInteractor, self).__init__(parent, title=title, size=(400, 700))

        self.categories = ['bear', 'camel', 'cat', 'cow', 'crocodile', 
                           'dog', 'elephant', 'flamingo', 'giraffe', 'hedgehog', 
                           'horse', 'kangaroo', 'lion', 'monkey', 'owl', 
                           'panda', 'penguin', 'pig', 'raccoon', 'rhinoceros', 
                           'sheep', 'squirrel', 'tiger', 'whale', 'zebra']
        self.categories_chinese = ['熊', '骆驼', '猫', '牛', '鳄鱼', 
                                   '狗', '象', '火烈鸟', '长颈鹿', '刺猬', 
                                   '马', '袋鼠', '狮子', '猴子', '猫头鹰', 
                                   '熊猫', '企鹅', '猪', '浣熊', '犀牛', 
                                   '羊', '松鼠', '老虎', '鲸鱼', '斑马']

        self.confusion_matrix = np.zeros((len(self.categories), len(self.categories)), dtype=np.int32)

        self.testSet = SketchDataset(
            mode='test',
            data_seq_dir='/home/purewhite/workspace/cs420/project/data/dataset_raw',
            data_img_dir=None,
            categories=self.categories,
            disable_augmentation=True
        )
        self.testLoader = DataLoader(self.testSet, batch_size=1, shuffle=True, pin_memory=False)
        self.it = iter(self.testLoader)

        self.img = None
        
        self.next()
        self.initUI()
    
    def next(self):
        cur = next(self.it)
        data = cur[0][0].numpy().astype(np.int32, 'C')
        self.curCategory = cur[-1].item()

        draw_strokes(data, './tmp.svg', width=224)
        with open('./tmp.svg', 'r') as f_in:
            svg = f_in.read()
            f_in.close()
        cairosvg.svg2png(bytestring=svg, write_to='./tmp.png')
        
        self.image_data = wx.Image('./tmp.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        
        if (self.img is not None):
            self.img.SetBitmap(self.image_data)
        
        print(f'Current: {self.categories_chinese[self.curCategory]} ({self.categories[self.curCategory]})')
    
    def initUI(self):
        panel = wx.Panel(self)
        
        self.rbox = wx.RadioBox(panel, label="选项 (Options)", choices=list(HumanInteractor.categoryValues.keys()), style=wx.RA_SPECIFY_ROWS)
        # self.rbox.Bind(wx.EVT_RADIOBOX, self.onRadioBoxChange)
        
        self.btn = wx.Button(panel, -1, "确认 (Confirm)", pos=(225, 100))
        self.btn.Bind(wx.EVT_BUTTON, self.onClicked)
        
        self.img = wx.StaticBitmap(panel, bitmap=self.image_data, pos=(160, 200))
        
        self.Center()
        self.Show(True)

    # def onRadioBoxChange(self, e):
    #     stringChosen = self.rbox.GetStringSelection()
    #     categoryChosen = HumanInteractor.categoryValues[stringChosen]
    #     print(stringChosen, categoryChosen)
    
    def onClicked(self, e):
        stringChosen = self.rbox.GetStringSelection()
        categoryChosen = HumanInteractor.categoryValues[stringChosen]
        print(f'Chosen: {stringChosen} {categoryChosen}')
        
        self.confusion_matrix[self.curCategory, categoryChosen] += 1
        # confusion_matrix[i, j]: True label is `i`, Chosen label is `j`
        self.next()
    
    def __del__(self):
        print(self.confusion_matrix)
        
        for k, i in HumanInteractor.categoryValues.items():
            TP = self.confusion_matrix[i, i]
            TN = np.sum(self.confusion_matrix[:i, :i]) + np.sum(self.confusion_matrix[:i, i+1:]) \
               + np.sum(self.confusion_matrix[i+1:, :i]) + np.sum(self.confusion_matrix[i+1:, i+1:])
    
            T = np.sum(self.confusion_matrix[i])    # = TP + FN
            P = np.sum(self.confusion_matrix[:, i]) # = TP + FP
            
            FN = T - TP
            FP = P - TP
            
            accuracy = (TP + TN) / (TP + FN + FP + TN)
            precision = TP / (TP + FP)
            recall = TP / (TP + TN)
            f1 = 2 * precision * recall / (precision + recall)
            
            print(f'{k}: Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1: {f1}')
        
        np.save('./result.npy', self.confusion_matrix)

ex = wx.App()
HumanInteractor(None)
ex.MainLoop()