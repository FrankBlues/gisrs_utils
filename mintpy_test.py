# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:32:46 2022

@author: DELL
"""

import os
import numpy as np
import matplotlib.pyplot as plt
# verify mintpy install
try:
    #from mintpy.objects.insar_vs_gps import plot_insar_vs_gps_scatter
    #from mintpy.unwrap_error_phase_closure import plot_num_triplet_with_nonzero_integer_ambiguity
    #from mintpy import workflow, view, tsview, plot_network, plot_transection, plot_coherence_matrix
    from mintpy import view, tsview
except ImportError:
    raise ImportError("Can not import mintpy!")

# utils function
def configure_template_file(outName, CONFIG_TXT): 
    """Write configuration files for MintPy to process HyP3 product"""
    if os.path.isfile(outName):
        with open(outName, "w") as fid:
            fid.write(CONFIG_TXT)
        print('write configuration to file: {}'.format(outName))

    else:
        with open(outName, "a") as fid:
            fid.write("\n" + CONFIG_TXT)
        print('add the following to file: \n{}'.format(outName))

# define the work directory
#work_dir = os.path.abspath(os.path.join(os.getcwd(), 'mintpy'))      #OpenSARLab at ASF
proj_name = 'timeline'
proj_dir = os.path.join(r'E:\S1', proj_name)  #Local
hyp3_dir = os.path.join(proj_dir, 'hyp3')
work_dir = os.path.join(proj_dir, 'mintpy')   #Local

if not os.path.isdir(proj_dir):
    os.makedirs(proj_dir)
    print('Create directory: {}'.format(proj_dir))
    
if not os.path.isdir(hyp3_dir):
    os.makedirs(hyp3_dir)
    print('Create directory: {}'.format(hyp3_dir))
    
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)
    print('Create directory: {}'.format(work_dir))
    
os.chdir(work_dir)
print('Go to work directory: {}'.format(work_dir))

CONFIG_TXT = f'''# vim: set filetype=cfg:
mintpy.load.processor        = hyp3
##---------interferogram datasets:
mintpy.load.unwFile          = {hyp3_dir}/*/*unw_phase_clip.tif
mintpy.load.corFile          = {hyp3_dir}/*/*corr_clip.tif
##---------geometry datasets:
mintpy.load.demFile          = {hyp3_dir}/*/*dem_clip.tif
mintpy.load.incAngleFile     = {hyp3_dir}/*/*lv_theta_clip.tif
mintpy.load.waterMaskFile    = {hyp3_dir}/*/*water_mask_clip.tif
'''
print(CONFIG_TXT)
configName = os.path.join(work_dir, "{}.txt".format(proj_name))
configure_template_file(configName, CONFIG_TXT)