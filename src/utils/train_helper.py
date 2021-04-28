import os
import numpy as np

def save_model_generic(epoch, save_top_dir, saver, sess):
    # save_top_dir = os.path.join('../saved_models', model_name)
    if not os.path.exists(save_top_dir):
        os.makedirs(save_top_dir)
    ckpt_path = os.path.join(save_top_dir,str(epoch) + 'model.ckpt')
    if epoch == 0:
        saver.save(sess, ckpt_path, write_meta_graph=True)
    else:
        saver.save(sess, ckpt_path, write_meta_graph=False)
    print("Saving model at epoch %d to %s" % (epoch, ckpt_path))

def restore_model_generic(epoch, save_top_dir, saver, sess):
    ckpt_path = os.path.join(save_top_dir,str(epoch)+'model.ckpt')
    print("restoring from %s" % ckpt_path)
    saver.restore(sess, ckpt_path)

            
def write_tb(loss_dict, writer, cat_name, total_ct):
    for loss_name, loss_val in loss_dict.items():
        writer.add_scalar('{}/{}'.format(cat_name, loss_name), loss_val, total_ct)

def loss_dict_to_str(loss_dict):
    ret_str = ''
    for loss_name in sorted(loss_dict):
        loss_val = loss_dict[loss_name]
        ret_str += '{} {:.4f} '.format(loss_name, loss_val)
    return ret_str

def loss_dict_obj_cat_to_str(loss_dict, result_file_name_per_object_cat=None):
    ret_arr = []
    for loss_name in sorted(loss_dict):
        ret_str = loss_name + ' '
        for obj_cat in sorted(loss_dict[loss_name]):
            if not (result_file_name_per_object_cat is None):
                n_tmp = len(result_file_name_per_object_cat[obj_cat])
                ret_str += '{} {:.3f}({}) '.format(obj_cat, loss_dict[loss_name][obj_cat], n_tmp)
            else:
                ret_str += '{} {:.3f}'.format(obj_cat, loss_dict[loss_name][obj_cat])

        ret_arr.append(ret_str)
    return ret_arr

def get_acc(gt_label, pred_cla):
    pos_label_idx = np.where(gt_label == 1)[0]
    neg_label_idx = np.where(gt_label == 0)[0]
    acc_cla_pos = np.mean(gt_label[pos_label_idx] == pred_cla[pos_label_idx])
    acc_cla_neg = np.mean(gt_label[neg_label_idx] == pred_cla[neg_label_idx])

    true_pos = np.sum(gt_label[pos_label_idx] == pred_cla[pos_label_idx])
    true_neg = np.sum(gt_label[neg_label_idx] == pred_cla[neg_label_idx])
    false_pos = np.sum(gt_label[neg_label_idx] != pred_cla[neg_label_idx])
    false_neg = np.sum(gt_label[pos_label_idx] != pred_cla[pos_label_idx])

    return {
        'acc': np.mean(gt_label == pred_cla),
        'acc_cla_pos': 1. * true_pos / (true_pos + false_neg + 1e-6),
        'acc_cla_neg': 1. * true_neg / (true_neg + false_pos + 1e-6),
        'precision': 1. * true_pos / (true_pos + false_pos + 1e-6),
        'recall': 1. * true_pos / (true_pos + false_neg + 1e-6),
    }   

import sys
import os
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'utils'))
sys.path.append(UTILS_DIR)
from data_helper import * 

class LossTracker:
    def __init__(self):
        self.loss_dict = {}
    
    def add(self, loss_val, loss_name):
        if not loss_name in self.loss_dict:
            self.loss_dict[loss_name] = []
        
        self.loss_dict[loss_name].append(loss_val)
    
    def add_dict(self, loss_dict):
        for loss_name in loss_dict:
            self.add(loss_dict[loss_name], loss_name)

    def stat(self):
        mean_dict = {}
        for loss_name in self.loss_dict:
            mean_dict[loss_name] = np.mean(self.loss_dict[loss_name])
        return mean_dict

    def reset(self):
        self.loss_dict = {}

def split_by_key(result_file_name_arr, key='object_cat'):
    ret_dict = {}
    assert key in ['object_cat', 'hook_name', 'object_name']
    for result_file_name in result_file_name_arr:
        if key == 'object_cat':
            _, _, tmp_key, _ = decode_result_file_name(result_file_name)
        elif key == 'hook_name':
            tmp_key, _ = split_result_file_name(result_file_name)
        elif key == 'object_name':
            _, tmp_key = split_result_file_name(result_file_name)
        if not (tmp_key in ret_dict):
            ret_dict[tmp_key] = []
        ret_dict[tmp_key].append(result_file_name)

    ret_dict_len = {tmp:len(ret_dict[tmp]) for tmp in ret_dict}
    return ret_dict, ret_dict_len
class LossTrackerMore():
    def __init__(self):
        self.loss_dict = {}
        self.loss_dict_result_file_name = {}
    
    def add(self, loss_val, loss_name, result_file_name):
        if not loss_name in self.loss_dict:
            self.loss_dict[loss_name] = []
        if not loss_name in self.loss_dict_result_file_name:
            self.loss_dict_result_file_name[loss_name] = []
        self.loss_dict[loss_name].append(loss_val)
        self.loss_dict_result_file_name[loss_name].append(result_file_name)
    
    def add_dict(self, loss_dict, result_file_name):
        for loss_name in loss_dict:
            self.add(loss_dict[loss_name], loss_name, result_file_name)

    def per_object_cat(self):
        loss_dict_ret = {}
        for loss_name in self.loss_dict:
            object_name_arr = [split_result_file_name(tmp)[1] for tmp in self.loss_dict_result_file_name[loss_name]]
            object_cat_arr = [split_name(tmp)[0] for tmp in object_name_arr]

            loss_dict_per_obj_cat = {}
            for i, object_cat in enumerate(object_cat_arr):
                if not object_cat in loss_dict_per_obj_cat:
                    loss_dict_per_obj_cat[object_cat] = []
                loss_dict_per_obj_cat[object_cat].append(self.loss_dict[loss_name][i])
            loss_dict_ret[loss_name] = loss_dict_per_obj_cat
        return loss_dict_ret
    def get_result_file_name_per_object_cat(self):
        all_result_file_names = self.loss_dict_result_file_name[list(self.loss_dict.keys())[0]]
        return split_by_key(all_result_file_names, key='object_cat')[0]

    def get_failed_result_file_names(self):
        failed_dict = {}
        for loss_name in self.loss_dict_result_file_name:
            idx = np.where(np.array(self.loss_dict[loss_name]) == 0)[0]
            failed_result_file_name = [self.loss_dict_result_file_name[loss_name][tmp] for tmp in idx]
            failed_dict[loss_name] = {
                'idx': idx,
                'result_file_name': failed_result_file_name
            }
        return failed_dict
    def stat(self):
        mean_dict = {}
        for loss_name in self.loss_dict:
            print(loss_name, np.sum(self.loss_dict[loss_name]), len(self.loss_dict[loss_name]))
            mean_dict[loss_name] = np.mean(self.loss_dict[loss_name])

        loss_dict_obj_cat = self.per_object_cat()
        mean_dict_obj_cat = {}
        for loss_name in loss_dict_obj_cat:
            mean_dict_obj_cat[loss_name] = {
                tmp:np.mean(loss_dict_obj_cat[loss_name][tmp]) for tmp in loss_dict_obj_cat[loss_name]
            }
        return mean_dict, mean_dict_obj_cat

    def print(self):
        mean_dict, mean_dict_obj_cat = self.stat()
        result_file_name_per_object_cat = self.get_result_file_name_per_object_cat()

        print(loss_dict_to_str(mean_dict)) 
    
        for tmp in loss_dict_obj_cat_to_str(mean_dict_obj_cat, result_file_name_per_object_cat):
            print(tmp)
    