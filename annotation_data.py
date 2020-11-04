import os
import sys
import csv
import numpy as np

from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common import utils

from tqdm import tqdm

# --------------------------------------------------------------------------------
class NuScene_Data_Creator:

    # --------------------------------------------------------------------------------
    def __init__(self, categories=['vehicle.car', 'vehicle.truck'],
                 attributes=['vehicle.parked', 'vehicle.stopped'],
                 ins_min_nr_anns=18):
        self.categories = categories
        self.attributes = attributes
        self.ins_min_nr_anns = ins_min_nr_anns

    # --------------------------------------------------------------------------------
    def create_data(self, nusc_path, nusc_type, save_path, verbose=True):
        nusc = NuScene_Data_Creator.create_nusc_obj(nusc_path=nusc_path, nusc_type=nusc_type, verbose=verbose)
        first_anns = self.get_first_ann(nusc=nusc)

        NuScene_Data_Creator.create_csv_files(nusc, first_anns, save_path)

    # --------------------------------------------------------------------------------
    @staticmethod
    def create_nusc_obj(nusc_path, nusc_type, verbose=True):
        if nusc_type is 'TrainVal':
            return NuScenes(version='v1.0-trainval', dataroot=nusc_path, verbose=verbose)
        elif nusc_type is 'Mini':
            return NuScenes(version='v1.0-mini', dataroot=nusc_path, verbose=verbose)
        else:
            print('\n--- Wrong type, use: TrainVal, Mini ---')
            sys.exit()

    # --------------------------------------------------------------------------------
    def get_first_ann(self, nusc):
        # Check if annotations are valid w.r.t to categories, attributes etc
        first_anns = []

        for instance in nusc.instance:
            category = nusc.get('category', instance['category_token'])
            nr_of_anns = instance['nbr_annotations']
            if category['name'] not in self.categories or nr_of_anns < self.ins_min_nr_anns:
                continue

            # Go through all annotations of that instance to check attribute
            current_ann = nusc.get('sample_annotation', instance['first_annotation_token'])

            is_valid = True
            while current_ann['next'] != '':
                if len(current_ann['attribute_tokens']) > 0:
                    attrib = nusc.get('attribute', current_ann['attribute_tokens'][0])['name']

                    if attrib in self.attributes:
                        is_valid = False
                        break
                else:
                    is_valid = False
                    break

                current_ann = nusc.get('sample_annotation', current_ann['next'])

            if is_valid:
                first_anns.append(instance['first_annotation_token'])

        return first_anns

    # --------------------------------------------------------------------------------
    @staticmethod
    def create_csv_files(nusc, anns, save_path):
        csv_columns = ['Instance_Token',
                       'Sample_Token',
                       'Ann_Token',
                       'Ann TS',
                       'Location',
                       'Pos_X', 'Pos_Y',
                       'Vel_X', 'Vel_Y',
                       'Acc_X', 'Acc_Y',
                       'Yaw',
                       'Pos_D_X', 'Pos_D_Y']


        print("\n--- Creating csv files for {} annotations ---".format(len(anns)))

        clean_files = []
        # Per Each Set of Annotations create csv file
        for ann in tqdm(anns):
            first_ann = nusc.get('sample_annotation', ann)

            # Scene ID and Location
            sample = nusc.get('sample', first_ann['sample_token'])
            scene = nusc.get('scene', sample['scene_token'])
            scene_id = int(scene['name'].replace('scene-', ''))
            location = nusc.get('log', scene['log_token'])['location']

            # Annotations
            prev_ann = first_ann
            current_ann = nusc.get('sample_annotation', first_ann['next']) # Start with 2nd annotation to derive acc
            next_ann  = nusc.get('sample_annotation', current_ann['next']) # Stop with 2nd last annotation to get poss_diff

            csv_path = os.path.join(save_path, 'Scene_{}_{}.csv'.format(scene_id, current_ann['token']))
            with open(csv_path, mode='w') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(csv_columns)

                ann_idx = 0
                # Body of a single csv
                while True:
                    # Tokens
                    instance_token = current_ann['instance_token']
                    sample_token = current_ann['sample_token']
                    ann_token = current_ann['token']

                    # X
                    t_0 = np.array([current_ann['translation']]).squeeze()
                    v_0 = np.array([nusc.box_velocity(current_ann['token'])]).squeeze()
                    a_0 = np.array([NuScene_Data_Creator.get_acceleration(nusc, current_ann, prev_ann)]).squeeze()
                    y_0 = np.array([utils.quaternion_yaw(Quaternion(current_ann['rotation']))])

                    # Y
                    t_1 = np.array([next_ann['translation']]).squeeze()
                    t_diff = t_1 - t_0

                    # Check for NaNs
                    if np.isnan(t_0).any() or np.isnan(v_0).any() or np.isnan(a_0).any() or np.isnan(y_0).any() or np.isnan(t_diff).any():
                        clean_files.append(csv_path)

                    if t_diff[0] == 0 or t_diff[1] == 0:
                        clean_files.append(csv_path)

                    # Data per sample
                    sample_data = [instance_token, sample_token, ann_token, ann_idx, location,
                                   t_0[0], t_0[1], v_0[0], v_0[1], a_0[0], a_0[1], y_0[0],
                                   t_1[0], t_1[1]]

                    csv_writer.writerow(sample_data)

                    # Break if no more annotations
                    if next_ann['next'] is '':
                        break

                    # Update anns
                    prev_ann = current_ann
                    current_ann = next_ann
                    next_ann = nusc.get('sample_annotation', current_ann['next'])
                    ann_idx += 1

        print('--- Cleaning Files ---')
        clean_files = set(clean_files)
        for file in clean_files:
            os.remove(file)

        print('--- Removed {} Files out of {} ---'.format(len(clean_files), len(anns)))
        print('--- Files saved at {} ---'.format(save_path))

    # --------------------------------------------------------------------------------
    @staticmethod
    def get_acceleration(nusc, current_ann, prev_ann):
        vel_init = np.array([nusc.box_velocity(prev_ann['token'])])
        vel_final = np.array([nusc.box_velocity(current_ann['token'])])

        ts_init = nusc.get('sample', prev_ann['sample_token'])['timestamp']
        ts_final = nusc.get('sample', current_ann['sample_token'])['timestamp']

        vel_delta = vel_final - vel_init

        ts_init = 1e-6 * ts_init
        ts_final = 1e-6 * ts_final

        ts_delta = ts_final - ts_init

        return vel_delta / ts_delta

    # --------------------------------------------------------------------------------
    @staticmethod
    def get_yaw_difference(nusc, current_ann, prev_ann):
        yaw_init = utils.quaternion_yaw(Quaternion(prev_ann['rotation']))
        yaw_final = utils.quaternion_yaw(Quaternion(current_ann['rotation']))

        ts_init = nusc.get('sample', prev_ann['sample_token'])['timestamp']
        ts_final = nusc.get('sample', current_ann['sample_token'])['timestamp']

        period = 2*np.pi
        yaw_delta = (yaw_final - yaw_init + period / 2) % period - period / 2
        if yaw_delta > np.pi:
            yaw_delta = yaw_delta - (2 * np.pi)

        ts_init = 1e-6 * ts_init
        ts_final = 1e-6 * ts_final

        ts_delta = ts_final - ts_init

        return yaw_delta / ts_delta
