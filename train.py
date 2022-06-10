import warnings
from run.cnn_train import SketchCNNTrain

if __name__ == '__main__':
    
    args = [
        '--model_fn', 'efficientnet_b0',
        
        '--data_seq_dir', '/home/purewhite/workspace/cs420/project/data/dataset_raw',
        '--data_img_dir', '/home/purewhite/workspace/cs420/project/data/dataset_processed_28',
        '--categories', str(['bear', 'cat', 'crocodile']),
        
        '--batch_size', str(8),
        '--num_epoch', str(2)
    ]
    
    with SketchCNNTrain(args) as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.run()