"""
Save NuScene annotation data to .csv files
"""

from annotation_data import NuScene_Data_Creator as ann_creator

# --------------------------------------------------------------------------------
if __name__ == '__main__':
    # NuScene paths
    train_val_path = 'nuScenes/TrainVal'
    mini_path = 'nuScenes/Mini'

    # CSV save paths
    train_val_save_path = 'TrainVal/'
    mini_save_path = 'Mini/'

    n = ann_creator()
    n.create_data(nusc_path=train_val_path,nusc_type='Mini',save_path=train_val_save_path)
    n.create_data(nusc_path=mini_path,nusc_type='TrainVal', save_path=mini_save_path)
