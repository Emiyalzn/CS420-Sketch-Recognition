#* For evaluation
import warnings
import os
from copy import deepcopy

# os.environ['CUDA_HOME'] = '/usr/local/cuda-11.2/'
# os.environ['CUDA_VISIBLE_DEVICES'] = 3

if __name__ == '__main__':
    from run.cnn_runner import SketchCNNRunner
    # efficientnet
    args = [
        '--model_fn', 'efficientnet_b0',
        # '--data_seq_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_seq_r2cnn',
        '--data_img_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_processed_28',
        '--batch_size', '48',
        '--seed', '[42,43,44]',
    ]
    with SketchCNNRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/efficientnet') as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.evaluate()
    
    # resnet
    args = [
        '--model_fn', 'resnet50',
        # '--data_seq_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_seq_r2cnn',
        '--data_img_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_processed_28',
        '--batch_size', '48',
        '--seed', '[42,43,44]',
    ]
    with SketchCNNRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/resnet') as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.evaluate()
            
    # sketchanet
    args = [
        '--model_fn', 'sketchanet',
        # '--data_seq_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_seq_r2cnn',
        '--data_img_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_processed_28',
        '--batch_size', '48',
        '--seed', '[42,43,44]',
    ]
    with SketchCNNRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/sketchanet') as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.evaluate()

    # r2cnn
    from run.r2cnn_runner import SketchR2CNNRunner
    # backbone: sketchanet
    args = [
        '--model_fn', 'sketchanet',
        '--data_seq_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_seq_r2cnn',
        # '--data_img_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_processed_28',
        '--batch_size', '64',
        '--seed', '[42]',
        '--intensity_channels', '8',
    ]
    with SketchR2CNNRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/r2cnn') as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.evaluate()
    
    # backbone: efficientnet_b0
    args = [
        '--model_fn', 'efficientnet_b0',
        '--data_seq_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_seq_r2cnn',
        # '--data_img_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_processed_28',
        '--batch_size', '64',
        '--seed', '[43,44]',
        '--intensity_channels', '8',
    ]
    with SketchR2CNNRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/r2cnn') as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.evaluate()


    from run.rnn_runner import SketchRNNRunner
    # gru
    args = [
        '--model_fn', 'gru',
        '--data_seq_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_seq_r2cnn',
        # '--data_img_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_processed_28',
        '--batch_size', '48',
        '--seed', '[42,43,44]',
    ]
    with SketchRNNRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/gru') as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.evaluate()
    # lstm
    args = [
        '--model_fn', 'lstm',
        '--data_seq_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_seq_r2cnn',
        # '--data_img_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_processed_28',
        '--batch_size', '48',
        '--seed', '[42,43,44]',
    ]
    with SketchRNNRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/lstm') as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.evaluate()


    from run.sketchmate_runner import SketchMateRunner
    # lstm
    args = [
        '--cnn_fn', 'efficientnet_b0',
        '--rnn_fn', 'lstm',
        '--data_seq_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_raw',
        '--data_img_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_processed_28',
        '--batch_size', '64',
        '--seed', '[42,43,44]',
    ]
    with SketchMateRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/sketchmate') as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.evaluate()
    

    from run.trans2cnn_runner import Trans2CNNRunner
    args = [
        '--model_fn', 'efficientnet_b0',
        '--dropout', '0.100',
        '--intensity_channels', '8',
        '--data_seq_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_seq_r2cnn',
        '--data_img_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_processed_28',
        '--batch_size', '128',
        '--seed', '[42,43,44]',
        '--do_reconstruction',
    ]
    # trans2cnn_full
    print('trans2cnn_full')
    with Trans2CNNRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/trans2cnn_full') as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.evaluate()
    # trans2cnn_woaug
    print('trans2cnn_woaug')
    with Trans2CNNRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/trans2cnn_woaug') as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.evaluate()
    # trans2cnn_woaug_worecon
    print('trans2cnn_woaug_worecon')
    with Trans2CNNRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/trans2cnn_woaug_worecon') as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.evaluate()

#* For visualization
# from run.cnn_runner import SketchCNNRunner
# from run.r2cnn_runner import SketchR2CNNRunner
# from run.rnn_runner import SketchRNNRunner
# from run.sketchmate_runner import SketchMateRunner
# from run.trans2cnn_runner import Trans2CNNRunner

# if __name__ == '__main__':
#     with SketchCNNRunner(local_dir='res/efficientnet') as app:
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')
#             app.visualize_emb()

#     with SketchR2CNNRunner(local_dir='res/r2cnn') as app:
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')
#             app.visualize_emb()

#     with SketchRNNRunner(local_dir="res/gru") as app:
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')
#             app.visualize_emb()

#     with SketchMateRunner() as app:
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')
#             app.visualize_emb()

#     with Trans2CNNRunner(local_dir='res/trans2cnn_full') as app:
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')
#             app.visualize_emb()

#* For robustness evaluation
# stroke_removal_probs = [0.0, 0.1, 0.2, 0.3, 0.4,
#                         0.5, 0.6, 0.7, 0.8, 0.9]
# stroke_deformation_settings = [(0.0, 00.0), (0.1, 10.0), (0.2, 20.0), (0.3, 30.0), (0.4, 40.0),
#                                (0.5, 50.0), (0.6, 60.0), (0.7, 70.0), (0.8, 80.0), (0.9, 90.0)]
#
# # gru
# from run.rnn_runner import SketchRNNRunner
#
# args = [
#     '--model_fn', 'gru',
#     '--data_seq_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_seq_r2cnn',
#     # '--data_img_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_processed_28',
#     '--batch_size', '48',
#     '--seed', '[44]',
# ]
# with SketchRNNRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/gru') as app:
#     with warnings.catch_warnings():
#         warnings.simplefilter('ignore')
#         app.robustness_experiment(stroke_removal_probs=stroke_removal_probs,
#                                   stroke_deformation_settings=stroke_deformation_settings)
#
# # r2cnn
# from run.r2cnn_runner import SketchR2CNNRunner
#
# # backbone: sketchanet
# args = [
#     '--model_fn', 'efficientnet_b0',
#     '--data_seq_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_seq_r2cnn',
#     # '--data_img_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_processed_28',
#     '--batch_size', '64',
#     '--seed', '[44]',
#     '--intensity_channels', '8',
# ]
# with SketchR2CNNRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/r2cnn') as app:
#     with warnings.catch_warnings():
#         warnings.simplefilter('ignore')
#         app.robustness_experiment(stroke_removal_probs=stroke_removal_probs,
#                                   stroke_deformation_settings=stroke_deformation_settings)
#
# # trans2cnn_full
# from run.trans2cnn_runner import Trans2CNNRunner
#
# args = [
#     '--model_fn', 'efficientnet_b0',
#     '--dropout', '0.100',
#     '--intensity_channels', '8',
#     '--data_seq_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_seq_r2cnn',
#     '--data_img_dir', '/home/lizenan/cs420/CS420-Proj/dataset/data/dataset_processed_28',
#     '--batch_size', '128',
#     '--seed', '[42]',
#     '--do_reconstruction',
# ]
# print('trans2cnn_full')
# with Trans2CNNRunner(args=args, local_dir='/home/lizenan/cs420/CS420-Proj/res/trans2cnn_full') as app:
#     with warnings.catch_warnings():
#         warnings.simplefilter('ignore')
#         app.robustness_experiment(stroke_removal_probs=stroke_removal_probs,
#                                   stroke_deformation_settings=stroke_deformation_settings)
