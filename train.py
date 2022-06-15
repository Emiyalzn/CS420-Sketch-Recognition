import warnings
from run.cnn_train import SketchCNNTrain

if __name__ == '__main__':
    
    args = [
        '--model_fn', 'efficientnet_b0',
        
        '--data_seq_dir', '~/cs420/dataset/data/dataset_raw',
        '--data_img_dir', '~/cs420/dataset/data/dataset_processed_224',
        '--categories', str(['dog', 'bear', 'kangaroo', 'whale', 'crocodile', 
                             'rhinoceros', 'penguin', 'camel', 'flamingo', 'giraffe', 
                             'pig', 'cat', 'cow', 'panda', 'lion', 
                             'tiger', 'raccoon', 'monkey', 'hedgehog', 'zebra', 
                             'horse', 'owl', 'elephant', 'squirrel', 'sheep']),
        
        '--batch_size', str(64),
        '--num_epoch', str(20),
        '--seed', str(42),
        
        '--disable_augmentation'
    ]
    
    with SketchCNNTrain(args) as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.run()