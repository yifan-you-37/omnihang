# Generative 3D Part Assembly via Dynamic Graph Learning

This is the implementation of ICRA 2021 paper "OmniHang: Learning to Hang Arbitrary Objects using Contact Point Correspondences and Neural Collision Estimation" created by 
Yifan You*, <a href="https://linsats.github.io/" target="_blank">Lin Shao*</a>, Toki Migimatsu, and <a href="https://web.stanford.edu/~bohg/" target="_blank">Jeannette Bohg</a>.

![image1](./images/teaser.png)

Hanging objects is a common daily task. Our system helps robots learn to hang arbitrary objects onto a diverse set of supporting items such as racks and hooks. All hanging poses rendered here are outputs of our proposed pipeline on object-supporting item pairs unseen during training.

- [paper link](https://arxiv.org/pdf/2103.14283.pdf)
- [project page](https://sites.google.com/view/hangingobject/)


<!-- ## File Structure

This repository provides data and code as follows.


```
    data/                       # contains PartNet data
        partnet_dataset/		# you need this dataset only if you  want to remake the prepared data
    prepare_data/				# contains prepared data you need in our exps 
    							# and codes to generate data
    	Chair.test.npy			# test data list for Chair (please download the .npy files using the link below)
    	Chair.val.npy			# val data list for Chair
    	Chair.train.npy 		# train data list for Chair
    	...
        prepare_shape.py				    # prepared data
    	prepare_contact_points.py			# prepared data for contact points
    	
    src/
    	utils/					# something useful
    	lin_my/	                # code for all training/testing/evaluation
            pointnet4/          # code adapted from PointNet++ (https://github.com/charlesq34/pointnet2)
            ...
            s1_train_matching.py # code for stage 1 training/evaluation
            s2a_train.py # code for stage 2a training/evaluation
            s2b_train_discrete.py # code for stage 2b training/evaluation
            s3_rl_collect.py # code for stage 3 RL online data collection. also used for stage 3 evaluation
            s3_rl_train.py # code for stage 3 RL online training
                
    		logs/				# contains checkpoints and tensorboard file
    		models/				# contains model file in our experiments
    		scripts/			# scrpits to train or test
    		data_dynamic.py		# code to load data
    		test_dynamic.py  	# code to test
    		train_dynamic.py  	# code to train
    		utils.py
    environment.yaml			# environments file for conda
    		

``` -->
<!-- 
This code has been tested on Ubuntu 16.04 with Cuda 10.0.130, GCC 7.5.0, Python 3.7.6 and PyTorch 1.1.0. 

Download the [pre-processed data](http://download.cs.stanford.edu/orion/genpartass/prepare_data.zip) for the .npy data files in file prepare_data/


## Dependencies

Please run
    

        conda env create -f environment.yaml
        . activate PartAssembly
        cd exps/utils/cd
        python setup.py build

to install the dependencies.

## Quick Start

Download [pretrained models](http://download.cs.stanford.edu/orion/genpartass/checkpoints.zip) and unzip under the root directory.

### Train the model

Simply run

        cd exps/dynamic_graph_learning/scripts/
        ./train_dynamic.sh
        
### Test the model

modify the path of the model in the test_dynamic.sh file

run

        cd exps/dynamic_graph_learning/scripts/
        ./test_dynamic.sh -->

## Questions

Please post issues for questions and more helps on this Github repo page. We encourage using Github issues instead of sending us emails since your questions may benefit others.

## Maintainers
@yifan-you-37 
@linsats


## Citation

  <!-- @InProceedings{HuangZhan2020PartAssembly,
      author = {Huang, Jialei and Zhan, Guanqi and Fan, Qingnan and Mo, Kaichun and Shao, Lin and Chen, Baoquan and Guibas, Leonidas and Dong, Hao},
      title = {Generative 3D Part Assembly via Dynamic Graph Learning},
      booktitle = {The IEEE Conference on Neural Information Processing Systems (NeurIPS)},
      year = {2020}
  } -->
    @misc{you2021omnihang,
      title={OmniHang: Learning to Hang Arbitrary Objects using Contact Point Correspondences and Neural Collision Estimation}, 
      author={Yifan You and Lin Shao and Toki Migimatsu and Jeannette Bohg},
      year={2021},
      eprint={2103.14283},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
    }

## License

MIT License

## Todos

Please request in Github Issue for more code to release.
