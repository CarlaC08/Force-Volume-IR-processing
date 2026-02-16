# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:15:01 2024

@author: collange
"""
#%% Packages

import streamlit as st
from st_flexible_callout_elements import flexible_success
from matplotlib.colors import *
import nanoscope as ns
from nanoscope import files
from nanoscope.constants import METRIC
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import itertools
from lmfit import Model
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import os as os
import ctypes
from ctypes import wintypes, windll
from pathlib import Path
from scipy.signal import savgol_filter
import win32api

#%% Variables
if 'original_path' not in st.session_state : st.session_state.original_path = os.getcwd()
if 'drive' not in st.session_state : st.session_state.drive = os.getcwd()
if 'root_path'  not in st.session_state : st.session_state.root_path = os.getcwd()
if 'selected_folder'  not in st.session_state : st.session_state.selected_folder = os.getcwd()
if 'selected_path' not in st.session_state : st.session_state.selected_path = ""
if 'width_px' not in st.session_state : st.session_state.width_px = 400
if 'height_px' not in st.session_state : st.session_state.height_px = 400
if 'x_select' not in st.session_state : st.session_state.x_select=0
if 'y_select' not in st.session_state : st.session_state.y_select=0
if 'fig_width' not in st.session_state : st.session_state.fig_width = 600
if 'fig_height' not in st.session_state : st.session_state.fig_height = 400
if 'FV_upload' not in st.session_state : st.session_state.FV_upload = False
if 'df_Peaks_ref' not in st.session_state : st.session_state.df_Peaks_ref = pd.DataFrame({'Peak n°' : 1, 'center' : 0., 'freq_min' : 0., 'freq_max' : 0., 'y_lim' : 0., 'f0_shift' : 0.5, 'FWHM_shift' : 10, 'window_freq' : 1.0, 'FWHM_threshold' : 30, 'per_integrale' : 3}, index=[1]).set_index('Peak n°')
if 'nbr_peak' not in st.session_state : st.session_state.nbr_peak = 1
if 'cube_Amp' not in st.session_state : st.session_state.cube_Amp = []
if 'cube_f0' not in st.session_state : st.session_state.cube_f0 = []
if 'cube_FWHM' not in st.session_state : st.session_state.cube_FWHM = []
if 'cube_FWHM_datas' not in st.session_state : st.session_state.cube_FWHM_datas = []
if 'cube_Damping' not in st.session_state : st.session_state.cube_Damping = []
if 'cube_B0' not in st.session_state : st.session_state.cube_B0 = []
if 'cube_Area_SHO' not in st.session_state : st.session_state.cube_Area_SHO = []
if 'cube_Area_datas' not in st.session_state : st.session_state.cube_Area_datas = []
if 'cube_x0' not in st.session_state : st.session_state.cube_x0 = []
if 'cube_center' not in st.session_state : st.session_state.cube_center = []
if 'cube_ymax' not in st.session_state : st.session_state.cube_ymax = []
if 'loaded_datas' not in st.session_state : st.session_state.loaded_datas = {"cube_Area_SHO_l":False,"cube_Amp_l":False,"cube_topo_l":True, "cube_Area_datas_l":False,"cube_ymax_l":False,"cube_center_l":False,"cube_FWHM_l":False, "cube_Damping_l":False,"cube_B0_l":False,"cube_x0_l":False,"cube_g0_l":False,"cube_Q_l":False,"cube_R2_l":False}
st.session_state.colorscales = [i for j in [[k, k+'_r'] for k in px.colors.named_colorscales()] for i in j]
st.session_state.default_colors = ['#000000', '#FF0000', '#0E00FF', '#06FF00', '#FB00FF', '#FFB300', '#00E0FF', '#8D00FF', "#0E8C8E", "#539032"]

#%% Fonctions
def toast_appearance():
    st.markdown(
        """
        <style>
            div[data-testid=toastContainer] {
                padding: 1% 4% 65% 2%;
                align-items: center;}
        
            div[data-testid=stToast] {
                padding: 20px 10px 40px 10px;
                margin: 10px 400px 200px 10px;
                background-color: #CECECE;
                width: 20%;}
            
            [data-testid=toastContainer] [data-testid=stMarkdownContainer] > p {
                font-size: 20px; font-style: normal; font-weight: 400;
                foreground-color: #ffffff;}
        </style>
        """, unsafe_allow_html=True)

def reset_datas():
    for key in ['cube_smoothed', 'cube_Amp', 'cube_f0', 'cube_FWHM', 'cube_Damping', 'cube_B0', 'cube_Area_SHO', 'cube_Area_datas', 'cube_x0', 'cube_center', 'cube_ymax', 'carto_offset', 'cube_FWHM_datas']:
        try : del st.session_state[key]
        except KeyError : pass

#from https://discuss.streamlit.io/t/simple-folder-explorer/77765
def GetDesktopFolder():
    CSIDL_DESKTOP = 0
    _SHGetFolderPath = windll.shell32.SHGetFolderPathW
    _SHGetFolderPath.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.HANDLE, wintypes.DWORD, wintypes.LPCWSTR]
    path_buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
    result = _SHGetFolderPath(0, CSIDL_DESKTOP, 0, 0, path_buf)
    return path_buf.value

def GetFolderList(root=None):
    if(root == None):
       root = os.path.expanduser('C:/Users/')
    subfolderObject = [x for x in os.scandir(root) ] 
    subfolders = [n.name for n in subfolderObject if n.is_dir() == True]
    cwd = os.getcwd()
    print(cwd)
    print("Subfolders(",cwd, ")", subfolders)
    return subfolders

@st.dialog("Select Folder")
def file_selector():
    st.info('Due to nanoscope limitation, accentuated character in the selected path may cause an error.')
    all_drives = win32api.GetLogicalDriveStrings().split('\000')[:-1]
    drive = st.radio('Choose a disk (default is your working directory)', [st.session_state.original_path]+all_drives, key='drive', on_change=new_rooth)

    root_path = st.session_state.root_path
    selected_folder = st.session_state.selected_folder
    subfolderObject = [x for x in os.scandir(root_path) ] 
    subfolders = [n.name for n in subfolderObject if n.is_dir() == True]
    subfolders.insert(0,"Parent")
    if "Desktop"not in subfolders:
        subfolders.insert(0,"Desktop")
    st.text_input("Root Folder", value=root_path)
    selected_index = 0
    pos = 0 
    for x in subfolders:
        if(st.button(x)):
            selected_index = pos
            if(selected_index == 0): # desktop
                st.session_state.root_path = GetDesktopFolder()
                st.session_state.selected_folder = ""
                subfolders.clear()
            elif(selected_index == 1): # parent
                st.session_state.root_path = Path(root_path).parent.absolute()
                st.session_state.selected_folder = ""
                subfolders.clear()
            else:
                if(st.session_state.selected_folder == subfolders[selected_index]):
                    st.session_state.root_path = os.path.join(root_path, selected_folder)
                    st.session_state.selected_folder = ""
                    subfolders.clear()
                else: st.session_state.selected_folder = subfolders[selected_index]
            st.rerun(scope="fragment")
        pos += 1
    button_label = "Select " + os.path.join(root_path, selected_folder)
    if(st.button(button_label, key="selectbutton", type="primary")):
        st.session_state.selected_path =  os.path.join(root_path, selected_folder)
        st.rerun()
  
def Config():
    if(st.button("Select Source Folder", key="SELECTFOLDER", on_click=reset_datas, type='primary')): file_selector()
    return st.session_state.selected_path

def select_file_cwd(wd):
    filenames = [i for i in os.listdir(wd) if i.endswith(".spm")]
    selected_filename = st.selectbox('Select a .spm file', filenames)
    return os.path.join(selected_filename)

def new_rooth():
    st.session_state.root_path = st.session_state.drive; st.session_state.selected_folder = st.session_state.drive
######

@st.cache_data(max_entries=1, show_spinner="Extracting data")
def extraction(filename):
    with files.ForceVolumeHoldFile(filename+'.spm') as file_:
        im_chan, data_chan_amp, data_chan_topo = file_.image_channel, file_[3],file_[2]
        n, m, l = im_chan.number_of_lines, im_chan.samples_per_line, data_chan_amp.number_of_hold_points_per_curve
        cube, cube_topo = np.zeros((n, m, l)), np.zeros((n, m, l))
        for i, j in itertools.product(np.arange(0,n),np.arange(0,m)):
            cube[i,j,:] = data_chan_amp.get_force_hold_data(i*m+j, METRIC)
            cube_topo[i,j,:] = data_chan_topo.get_force_hold_data(i*m+j, METRIC)
        x, aspect_ratio = im_chan.scan_size, im_chan.shape[0]/im_chan.shape[1]
        y, xy_unit = x * aspect_ratio, im_chan.scan_size_unit
        chan_name, fwt, f_range = data_chan_amp.data_type_desc, data_chan_amp.force_sweep_type, data_chan_amp.force_sweep_freq_range
        frequencies=np.linspace(f_range[0]*10**-3, f_range[1]*10**-3, l)
        return n, m, l, x, y, xy_unit, chan_name, fwt, frequencies, cube, -cube_topo[:,:,0]

@st.cache_data(max_entries=1, show_spinner="Smoothing all the frequency spectra")
def smoothing(amplitude, freq, window_length, polyorder, offset_type, offset_values):
    Y = savgol_filter(amplitude, window_length, polyorder)
    if offset_type=='Range selection' : bkg = np.mean(amplitude[(freq >= offset_values[0]) & (freq <= offset_values[1])])
    elif offset_type=='Input personnalised value' : bkg = offset_values
    elif offset_type=='None' : bkg = 0
    return Y-bkg, bkg

def smoothing_SG(cube, frequencies, window_length, polyorder, n, m, l, offset_type, offset_values):
    parallel = Parallel(n_jobs=-1, return_as="generator")
    output_generator = parallel(delayed(smoothing)(cube[i, j], frequencies, window_length, polyorder, offset_type, offset_values) for i in range(n) for j in range(m))
    res = list(output_generator)
    cube_smoothed, carto_offset = np.asarray([i[0] for i in res]).reshape(n,m,l), np.asarray([i[1] for i in res]).reshape(n,m)
    return cube_smoothed, carto_offset

def save_datas(savename, datas_to_save):
    with open(savename, 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(datas_to_save.shape))
        for data_slice in datas_to_save:
            np.savetxt(outfile, data_slice, fmt='%-10.8f')
            outfile.write('# New slice\n')

#%% Fonctions fit SHO asymetric
def SHO_asym(B0, f, x0, f0, D):
    return B0/np.sqrt((((2*np.pi*f)-(2*np.pi*x0))**2-(2*np.pi*f0)**2)**2+(D*((2*np.pi*f)-(2*np.pi*x0)))**2)

def SHO_asym_Fit(frequency, amplitude, min_freq, max_freq, ylim, window_freq, f0_shift, FWHM_shift, i, j):
    half_window_freq, y, x = window_freq/2, amplitude[(frequency >= min_freq) & (frequency <= max_freq)], frequency[(frequency >= min_freq) & (frequency <= max_freq)]
    b=np.where(y==y.max())[0][0] # Find the max
    if y[b]>ylim:
        condition = (frequency >= x[b]-half_window_freq) & (frequency <= x[b]+half_window_freq)
        model = Model(SHO_asym, independent_vars=['f'], nan_policy='raise')
        x_fitwind, y_fitwind = frequency[condition], amplitude[condition]
        try : no_fit_FWHM(x_fitwind, y_fitwind)
        except ValueError : D, FWHM_initial, FWHM, amp_max, center, B0, f0, x0, R2 = np.nan, np.nan, np.nan, np.nan, x[b], np.nan, np.nan, np.nan, np.nan
        else :
            f_maxHW, f_minHW = no_fit_FWHM(x_fitwind, y_fitwind)
            FWHM_initial = f_maxHW - f_minHW
            if y_fitwind[0]<y_fitwind[-1] : params = model.make_params(B0={'value':1e5, 'min':0}, x0={'value':2*x[b], 'expr':'2*f0 '}, f0={'value':x[b], 'min':x[b]-f0_shift, 'max':x[b]+f0_shift}, D={'value': 2*np.pi*FWHM_initial/np.sqrt(3), 'min':(2*np.pi/np.sqrt(3))*(1-(FWHM_shift/100))*(FWHM_initial), 'max':(2*np.pi/np.sqrt(3))*(1+(FWHM_shift/100))*(FWHM_initial)})
            elif y_fitwind[0]>y_fitwind[-1] : params = model.make_params(B0={'value':1e5, 'min':0}, x0={'value':0, 'min':0, 'vary':False}, f0={'value':x[b], 'min':x[b]-f0_shift, 'max':x[b]+f0_shift}, D={'value': 2*np.pi*FWHM_initial/np.sqrt(3), 'min':(2*np.pi/np.sqrt(3))*(1-(FWHM_shift/100))*(FWHM_initial), 'max':(2*np.pi/np.sqrt(3))*(1+(FWHM_shift/100))*(FWHM_initial)})
            else : params = model.make_params(B0={'value':1e5, 'min':0}, x0={'value':0, 'min':0}, f0={'value':x[b], 'min':x[b]-f0_shift, 'max':x[b]+f0_shift}, D={'value': 2*np.pi*FWHM_initial/np.sqrt(3), 'min':(2*np.pi/np.sqrt(3))*(1-(FWHM_shift/100))*(FWHM_initial), 'max':(2*np.pi/np.sqrt(3))*(1+(FWHM_shift/100))*(FWHM_initial)})
            out = model.fit(y_fitwind, params, f=x_fitwind)
            B0, D, f0, x0 = out.summary()['best_values']['B0'], out.summary()['best_values']['D'], out.summary()['best_values']['f0'], out.summary()['best_values']['x0']
            f_maxHW, f_minHW = no_fit_FWHM(x_fitwind, out.best_fit)
            FWHM = f_maxHW - f_minHW
            amp_max = max(SHO_asym(B0, x_fitwind, x0, f0, D))
            center = x_fitwind[SHO_asym(B0, x_fitwind, x0, f0, D)==amp_max][0]
            R2 = out.rsquared
    else : D, FWHM_initial, FWHM, amp_max, center, B0, f0, x0, R2 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    return D, FWHM_initial, FWHM, amp_max, center, B0, f0, x0, y[b], R2

def SHO_asym_integrale(B0, D, f0, x0, center, frequency, amplitude, half_int_wind, i, j):
    condition = (frequency>=center-half_int_wind)&(frequency<=center+half_int_wind)
    area_SHO = np.trapz(SHO_asym(B0, frequency[condition], x0, f0, D), frequency[condition])
    area_datas = np.trapz(amplitude[condition], frequency[condition])
    if B0==0: area_SHO=np.nan
    return area_SHO, area_datas

@st.cache_data(max_entries=20, show_spinner="Fitting the asymetric SHO to all the position")
def SHO_asym_parameters(frequency, amplitude, n, m, freq_min, freq_max, ylim, f0_shift, FWHM_shift, window_freq, ncores):
    parallel = Parallel(n_jobs=ncores, return_as="generator", verbose=1)
    output_generator = parallel(delayed(SHO_asym_Fit)(frequency, amplitude[i, j], freq_min, freq_max, ylim, window_freq, f0_shift, FWHM_shift, i, j) for i in range(n) for j in range(m))
    res=list(output_generator)
    cube_Damping, cube_FWHM_init, cube_FWHM, cube_Amp, cube_center, cube_B0, cube_f0, cube_x0, cube_ymax, cube_R2 = np.asarray(res).T
    cube_Damping, cube_FWHM_init, cube_FWHM, cube_Amp, cube_center, cube_f0, cube_x0, cube_B0, cube_ymax, cube_R2 = cube_Damping.reshape(n, m), cube_FWHM_init.reshape(n, m), cube_FWHM.reshape(n, m), cube_Amp.reshape(n, m), cube_center.reshape(n, m), cube_f0.reshape(n, m), cube_x0.reshape(n, m), cube_B0.reshape(n, m), cube_ymax.reshape(n, m), cube_R2.reshape(n, m)
    return cube_Amp, cube_center, cube_x0, cube_FWHM, cube_FWHM_init, cube_Damping, cube_B0, cube_f0, cube_ymax, cube_R2

@st.cache_data(max_entries=20, show_spinner="Computing the area to all the position")
def area_asym_computing(frequency, amplitude, n, m, cube_B0, cube_Damping, cube_FWHM, cube_f0, cube_x0, cube_center, FWHM_threshold, per_integrale):
    FWHM_max = np.nanmax(cube_FWHM[cube_FWHM<=FWHM_threshold])
    k, l = np.asarray(np.where(cube_FWHM==FWHM_max), dtype=int)
    k,l = k[0], l[0]
    st.write("FWHM damping :", np.round(FWHM_max,2),"at coordinates : n=",k,', m=',l)
    integration_window = per_integrale*cube_FWHM[k,l]
    st.write('Integration window =',np.round(integration_window,2),' kHz')
    half_int_wind = integration_window/2
    parallel = Parallel(n_jobs=-1, return_as="generator", verbose=1)
    output_generator = parallel(delayed(SHO_asym_integrale)(cube_B0[i, j], cube_Damping[i, j], cube_f0[i, j], cube_x0[i, j], cube_center[i, j], frequency, amplitude[i, j], half_int_wind, i, j) for i in range(n) for j in range(m))
    res=list(output_generator)
    cube_Area_SHO, cube_Area_datas = np.asarray(res).T
    cube_Area_SHO, cube_Area_datas = cube_Area_SHO.reshape(n, m), cube_Area_datas.reshape(n, m)
    return cube_Area_SHO, cube_Area_datas, integration_window

def SHO_asym_plot(frequency, amplitude, test_choice, center, freq_min, freq_max, y_lim, f0_shift, FWHM_shift, window_freq, FWHM_threshold, per_integrale):
    half_window_freq, y, x = window_freq/2, amplitude[(frequency >= freq_min) & (frequency <= freq_max)], frequency[(frequency >= freq_min) & (frequency <= freq_max)]
    b=np.where(y==y.max())[0][0]
    if y[b] < y_lim : st.write("The maximal amplitude is strictly inferior to your amplitude threshold. Choose an other position.")
    else :
        model = Model(SHO_asym, independent_vars=['f'], nan_policy='raise')
        try : no_fit_FWHM(x, y)
        except ValueError : st.write(f'No peak found between {freq_min} and {freq_max} kHz.')
        else :
            f_maxHW, f_minHW = no_fit_FWHM(x, y)
            FWHM_init = f_maxHW - f_minHW
            condition = (frequency>=x[b]-half_window_freq) & (frequency<=x[b]+half_window_freq)
            if amplitude[condition][0]<amplitude[condition][-1] : params = model.make_params(B0={'value':1e5, 'min':0}, x0={'value':2*x[b], 'expr':'2*f0 '}, f0={'value':x[b], 'min':x[b]-f0_shift, 'max':x[b]+f0_shift}, D={'value': 2*np.pi*FWHM_init/np.sqrt(3), 'min':(2*np.pi/np.sqrt(3))*(1-(FWHM_shift/100))*(FWHM_init), 'max':(2*np.pi/np.sqrt(3))*(1+(FWHM_shift/100))*(FWHM_init)})
            else : params = model.make_params(B0={'value':1e5, 'min':0}, x0={'value':0, 'min':0, 'vary':False}, f0={'value':x[b], 'min':x[b]-f0_shift, 'max':x[b]+f0_shift}, D={'value': 2*np.pi*FWHM_init/np.sqrt(3), 'min':(2*np.pi/np.sqrt(3))*(1-(FWHM_shift/100))*(FWHM_init), 'max':(2*np.pi/np.sqrt(3))*(1+(FWHM_shift/100))*(FWHM_init)})
            out = model.fit(amplitude[condition], params, f=frequency[condition])
            B0, D, f0, x0 = out.summary()['best_values']['B0'], out.summary()['best_values']['D'], out.summary()['best_values']['f0'], out.summary()['best_values']['x0']
            y_fit = out.best_fit
            f_maxHW, f_minHW = no_fit_FWHM(frequency[condition], y_fit)
            FWHM = f_maxHW - f_minHW
            integral_window=FWHM*per_integrale
            fig = go.Figure(layout=dict(height=500, width=900))
            fig.add_traces(go.Scatter(x=frequency, y=amplitude, mode='lines', name=test_choice,legendrank=1))
            fig.add_traces(go.Scatter(x=np.arange(min(frequency),max(frequency),0.01), y=SHO_asym(B0, np.arange(min(frequency),max(frequency),0.01), x0, f0, D), mode='lines', name='Asymetric SHO fit on all f with df = 0.01 kHz'))
            fig.add_traces(go.Scatter(x=frequency[condition], y=y_fit, mode='markers', name='Fit',legendrank=3))
            fig.add_shape(type="rect", xref="x", yref="paper", x0=min(frequency[condition]), legendrank=2, y0=0, x1=max(frequency[condition]), y1=np.max(amplitude), showlegend=True, line=dict(color="black", width=2), fillcolor='black', name='Fit window', opacity=0.1)
            fig.add_traces(go.Scatter(x=frequency[(frequency>=(f0-integral_window/2))&(frequency<=(f0+integral_window/2))], y=SHO_asym(B0, frequency[(frequency>=f0-(integral_window/2))&(frequency<=f0+(integral_window/2))], x0, f0, D), fill='tozeroy', marker={'opacity': 0}, mode='markers', name='Integration range'))
            fig.update_yaxes(title='Amplitude (mV)', range=[0, np.max(amplitude)])
            fig.update_xaxes(title='Frequency (kHz)', range=[freq_min-window_freq, freq_max+window_freq])
            st.plotly_chart(fig)
            st.write(f"R² = {out.rsquared}")
            st.write(f'Initial FWHM = {np.round(FWHM_init,2)} kHz')
            st.write(f'ASHO FWHM = {np.round(FWHM,2)} kHz')
            st.write(f'Integration window = {np.round(integral_window,2)} kHz')
            st.write(f'Area of the asymetric SHO calculated on the integration range = {np.round(np.trapz(SHO_asym(B0, frequency[(frequency>=f0-(integral_window/2))&(frequency<=f0+(integral_window/2))], x0, f0, D), frequency[(frequency>=f0-(integral_window/2))&(frequency<=f0+(integral_window/2))]), 2)} mV.kHz')

#%% Fonctions fit SHO
def SHO(B0, D, f0, f):
    return B0/np.sqrt(((2*np.pi*f0)**2-(2*np.pi*f)**2)**2+(D*(2*np.pi*f))**2)

def SHO_Fit(frequency, amplitude, min_freq, max_freq, ylim, f0_shift, FWHM_shift, window_freq, i, j):
    half_window_freq, y, x = window_freq/2, amplitude[(frequency >= min_freq) & (frequency <= max_freq)], frequency[(frequency >= min_freq) & (frequency <= max_freq)]
    b=np.argmax(y) # Find the max
    if y[b]>ylim:
        condition = (frequency >= x[b]-half_window_freq) & (frequency <= x[b]+half_window_freq)
        x_fitwind, y_fitwind = frequency[condition], amplitude[condition]
        try : no_fit_FWHM(x_fitwind, y_fitwind)
        except ValueError : D, FWHM_init, amp_max, center, B0, R2 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else :
            f_maxHW, f_minHW = no_fit_FWHM(x, y)
            FWHM_init = f_maxHW - f_minHW
            model = Model(SHO, independent_vars=['f'], nan_policy='raise')
            params = model.make_params(B0={'value':y[b], 'min':0}, D={'value': 2*np.pi*FWHM_init/np.sqrt(3), 'min':(2*np.pi/np.sqrt(3))*(1-(FWHM_shift/100))*(FWHM_init), 'max':(2*np.pi/np.sqrt(3))*(1+(FWHM_shift/100))*(FWHM_init)}, f0={'value':x[b], 'min':x[b]-f0_shift, 'max':x[b]+f0_shift})
            out = model.fit(y_fitwind, params, f=x_fitwind)
            B0, D, center, R2 = out.summary()['best_values']['B0'], out.summary()['best_values']['D'], out.summary()['best_values']['f0'], out.rsquared
            amp_max = max(SHO(B0, D, center, x_fitwind))
    else: D, FWHM_init, amp_max, center, B0, R2 = np.nan, np.nan, x[b], np.nan, np.nan
    return D, FWHM_init, amp_max, center, B0, y[b], R2

def SHO_integrale(B0, D, f0, frequency, amplitude, half_int_wind, i, j):
    reduc_freq = frequency[(frequency>=f0-half_int_wind)&(frequency<=f0+half_int_wind)]
    area_SHO = np.trapz(SHO(B0, D, f0, reduc_freq), reduc_freq)
    area_datas = np.trapz(amplitude[(frequency>=f0-half_int_wind)&(frequency<=f0+half_int_wind)], reduc_freq)
    if B0==0: area_SHO=np.nan
    return area_SHO, area_datas

def SHO_plot(frequency, amplitude, test_choice, center, freq_min, freq_max, y_lim, f0_shift, FWHM_shift, window_freq, FWHM_threshold, per_integrale):
    half_window_freq, y, x = window_freq/2, amplitude[(frequency >= freq_min) & (frequency <= freq_max)], frequency[(frequency >= freq_min) & (frequency <= freq_max)]
    b=np.where(y==y.max())[0][0] # Find the max
    if y[b] < y_lim : st.write("The maximal amplitude is strictly inferior to your amplitude threshold. Choose an other position.")
    else :
        condition = (frequency>=x[b]-half_window_freq) & (frequency<=x[b]+half_window_freq)
        model = Model(SHO, independent_vars=['f'], nan_policy='raise')
        try : no_fit_FWHM(x, y)
        except ValueError : st.write(f'No peak found between {freq_min} and {freq_max} kHz.')
        else :
            f_maxHW, f_minHW = no_fit_FWHM(x, y)
            FWHM_init = f_maxHW - f_minHW
            params = model.make_params(B0={'value':y[b], 'min' :0}, D={'value': 2*np.pi*FWHM_init/np.sqrt(3), 'min':(2*np.pi/np.sqrt(3))*(1-(FWHM_shift/100))*(FWHM_init), 'max':(2*np.pi/np.sqrt(3))*(1+(FWHM_shift/100))*(FWHM_init)}, f0={'value':x[b], 'min':x[b]-f0_shift, 'max':x[b]+f0_shift})
            out = model.fit(amplitude[condition], params, f=frequency[condition])
            y_fit = out.best_fit
            B0, D, f0 = out.summary()['best_values']['B0'], out.summary()['best_values']['D'], out.summary()['best_values']['f0']  
            FWHM = D*np.sqrt(3)/(2*np.pi)
            integral_window = per_integrale*FWHM
            fig = go.Figure(layout=dict(height=500, width=900))
            fig.add_traces(go.Scatter(x=frequency, y=amplitude, mode='lines', name=test_choice, legendrank=1))
            fig.add_traces(go.Scatter(x=np.arange(min(frequency),max(frequency),0.01), y= SHO(B0, D, f0, np.arange(min(frequency),max(frequency),0.01)), mode='lines', name='SHO fit on all f with df = 0.01 kHz'))
            fig.add_traces(go.Scatter(x=frequency[condition], y=y_fit, mode='markers', name='Fit',legendrank=3))
            fig.add_shape(type="rect", xref="x", yref="paper", x0=min(frequency[condition]), legendrank=2, y0=0, x1=max(frequency[condition]), y1=np.max(amplitude), showlegend=True, line=dict(color="black", width=2), fillcolor='black', name='Fit window', opacity=0.1)
            fig.add_traces(go.Scatter(x=frequency[(frequency>=(f0-integral_window/2))&(frequency<=(f0+integral_window/2))], y=SHO(B0, D, f0, frequency[(frequency>=(f0-integral_window/2))&(frequency<=(f0+integral_window/2))]), fill='tozeroy', marker={'opacity': 0}, mode='markers', name='Integration range'))
            fig.update_yaxes(title='Amplitude (mV)', range=[0, np.max(amplitude)])
            fig.update_xaxes(title='Frequency (kHz)', range=[freq_min-window_freq, freq_max+window_freq])
            st.plotly_chart(fig)
            st.write(f"R² = {out.rsquared}")
            st.write(f'Initial FWHM = {np.round(FWHM_init,2)} kHz')
            st.write(f'SHO FWHM = {np.round(FWHM,2)} kHz')
            st.write(f'Integration window = {np.round(integral_window,2)} kHz')
            st.write(f'Area of the SHO calculated on the integration range = {np.round(np.trapz(SHO(B0, D, f0, frequency[(frequency>=(f0-integral_window/2))&(frequency<=(f0+integral_window/2))]), frequency[(frequency>=(f0-integral_window/2))&(frequency<=(f0+integral_window/2))]), 2)} mV.kHz')

@st.cache_data(max_entries=1, show_spinner="Fitting the SHO to all the position")
def SHO_parameters(x, y, n, m, freq_min, freq_max, y_lim, f0_shift, FWHM_shift, half_window_freq, ncores):
    parallel = Parallel(n_jobs=ncores, return_as="generator", verbose=1)
    output_generator = parallel(delayed(SHO_Fit)(x, y[i, j], freq_min, freq_max, y_lim, f0_shift, FWHM_shift, half_window_freq, i, j) for i in range(n) for j in range(m))
    res=list(output_generator)
    cube_Damping, cube_FWHM_init, cube_Amp, cube_f0, cube_B0, cube_ymax, cube_R2 = np.asarray(res).T
    cube_Damping, cube_FWHM_init, cube_Amp, cube_f0, cube_B0, cube_ymax, cube_R2 = cube_Damping.reshape(n, m), cube_FWHM_init.reshape(n, m), cube_Amp.reshape(n, m), cube_f0.reshape(n, m), cube_B0.reshape(n, m), cube_ymax.reshape(n, m), cube_R2.reshape(n, m)
    cube_FWHM = np.sqrt(3)/(2*np.pi)*cube_Damping
    return cube_Amp, cube_f0, cube_FWHM, cube_FWHM_init, cube_Damping, cube_B0, cube_ymax, cube_R2

@st.cache_data(max_entries=1, show_spinner="Computing the area to all the position")
def area_computing(frequencies, amplitude, n, m, cube_B0, cube_Damping, cube_FWHM, cube_f0, FWHM_threshold, per_integrale):
    FWHM_max = np.nanmax(cube_FWHM[cube_FWHM<=FWHM_threshold])
    k, l = np.asarray(np.where(cube_FWHM==FWHM_max), dtype=int)
    k, l = k[0], l[0]
    st.write(f"Max FWHM :{np.round(FWHM_max,2)} at coordinates : n={k}, m={l}")
    integration_window = per_integrale*cube_FWHM[k,l]
    st.write(f'Integration window = {np.round(integration_window,2)} kHz')
    parallel = Parallel(n_jobs=-1, return_as="generator", verbose=1)
    output_generator = parallel(delayed(SHO_integrale)(cube_B0[i, j], cube_Damping[i, j], cube_f0[i, j], frequencies, amplitude[i, j], integration_window/2, i, j) for i in range(n) for j in range(m))
    res=list(output_generator)
    cube_Area_SHO, cube_Area_datas = np.asarray(res).T
    cube_Area_SHO, cube_Area_datas = cube_Area_SHO.reshape(n, m), cube_Area_datas.reshape(n, m)
    return cube_Area_SHO, cube_Area_datas, integration_window

#%% Fonctions no fit

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def find_sym_nearest(array, value): #find the two nearest values of a symmetric curve
    imax = np.argmax(array)
    y1, idx1 = find_nearest(array[:imax], value)
    y2, idx2 = find_nearest(array[imax:], value)
    return ((y1, idx1), (y2, imax + idx2))

def no_fit_FWHM(Xarray, Yarray):
    YMAX = np.max(Yarray)
    (y1, idx1), (y2, idx2) = find_sym_nearest(Yarray, YMAX/2)    
    return Xarray[idx2], Xarray[idx1]

def no_fit(frequencies, cube, n, m, freq_min, freq_max, y_lim):
    f_minprox, i_minprox = find_nearest(frequencies, freq_min)
    f_maxprox, i_maxprox = find_nearest(frequencies, freq_max)
    x = frequencies[i_minprox:i_maxprox]
    f_interp = np.arange(f_minprox, f_maxprox, 0.1)
    cube_FWHM, cube_amp, cube_f0 = np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, m))
    for i, j in itertools.product(range(0,n,1), range(0,m,1)):
        Ycube = cube[i, j, i_minprox:i_maxprox]
        Ycube_interp = np.interp(f_interp, x, Ycube)        
        try : no_fit_FWHM(f_interp, Ycube_interp)
        except ValueError : cube_FWHM[i, j], cube_amp[i, j], cube_f0[i, j] = np.nan, np.nan, np.nan
        else :
            f_maxHW, f_minHW = no_fit_FWHM(f_interp, Ycube_interp)
            cube_FWHM[i, j] = f_maxHW - f_minHW
            cube_amp[i, j] = np.max(Ycube)
            i_max = np.argmax(Ycube) 
            cube_f0[i, j] = frequencies[i_minprox + i_max]
    return cube_FWHM, cube_amp, cube_f0
    
def area_computing_nofit(frequency, amplitude, n, m, cube_FWHM, cube_f0, FWHM_threshold, per_integrale):
    FWHM_max = np.nanmax(cube_FWHM[cube_FWHM<=FWHM_threshold])
    k, l = np.asarray(np.where(cube_FWHM==FWHM_max), dtype=int)
    k,l = k[0], l[0]
    st.write(f"Max FWHM = {np.round(FWHM_max,2)} at coordinates : n={k}, m={l}")
    integration_window = per_integrale*cube_FWHM[k,l]
    half_int_wind = integration_window/2
    st.write(f'Integration window = {np.round(integration_window,2)} kHz')
    cube_Area_datas = np.zeros((n,m))
    for i, j in itertools.product(range(0,n,1), range(0,m,1)):
        if np.isnan(cube_FWHM[i,j]) == True : cube_Area_datas[i,j] = np.nan
        else :
            f0 = cube_f0[i,j]
            cube_Area_datas[i,j] = np.trapz(amplitude[i,j][(frequency>=f0-half_int_wind)&(frequency<=f0+half_int_wind)], frequency[(frequency>=f0-half_int_wind)&(frequency<=f0+half_int_wind)])
    return cube_Area_datas, integration_window

def nofit_plot(frequency, amplitude, test_choice, center, freq_min, freq_max, y_lim, f0_shift, FWHM_shift, window_freq, FWHM_threshold, per_integrale):
    y_search = amplitude[(frequency >= freq_min) & (frequency <= freq_max)]
    b=np.where(y_search==y_search.max())[0][0] # Find the max
    if y_search[b] < y_lim : st.write("The maximal amplitude is strictly inferior to your amplitude threshold. Choose an other position.")
    else :
        f_minprox, i_minprox = find_nearest(frequency, freq_min)
        f_maxprox, i_maxprox = find_nearest(frequency, freq_max)
        x = frequency[i_minprox:i_maxprox]
        y = amplitude[i_minprox:i_maxprox]
        try : no_fit_FWHM(x, y)
        except ValueError : st.write(f'No peak found between {freq_min} and {freq_max} kHz.')
        else :
            f_interp = np.arange(f_minprox, f_maxprox, 0.1)
            Y_interp = np.interp(f_interp, x, y)        
            f_maxHW, f_minHW = no_fit_FWHM(f_interp, Y_interp)
            FWHM = f_maxHW - f_minHW
            i_max = np.argmax(y) 
            f0 = frequency[i_minprox + i_max]
            FWHM = f_maxHW - f_minHW
            integral_window = per_integrale*FWHM
            half_int_wind = integral_window/2
            area = np.trapz(amplitude[(frequency>=f0-half_int_wind)&(frequency<=f0+half_int_wind)], frequency[(frequency>=f0-half_int_wind)&(frequency<=f0+half_int_wind)])
            fig = go.Figure(layout=dict(height=500, width=900))
            fig.add_traces(go.Scatter(x=frequency, y=amplitude, mode='lines', name=test_choice, legendrank=1))
            fig.add_traces(go.Scatter(x=f_interp, y=Y_interp, mode='lines', name='Interpolation', legendrank=1))
            fig.add_traces(go.Scatter(x=frequency[(frequency>=f0-half_int_wind)&(frequency<=f0+half_int_wind)], y=amplitude[(frequency>=f0-half_int_wind)&(frequency<=f0+half_int_wind)], fill='tozeroy', marker={'opacity': 0}, mode='markers', name='Integration range'))
            fig.update_yaxes(title='Amplitude (mV)', range=[0, np.max(amplitude)])
            fig.update_xaxes(title='Frequency (kHz)', range=[freq_min-window_freq, freq_max+window_freq])
            st.plotly_chart(fig)
            st.write(f'FWHM = {np.round(FWHM,2)} kHz')
            st.write(f'Integration window = {np.round(integral_window, 2)} kHz')
            st.write(f'Area of the SHO calculated on the integration range = {np.round(area, 2)} mV.kHz')

#%% Fonctions plot
@st.cache_data(max_entries=1, show_spinner=False)
def topo_plot(cube_topo, color_topo_map, x_select, y_select, c_max, c_min, width_px, height_px) :
    fig = go.Figure(layout=dict(title='Topography map', height=height_px, width=width_px, xaxis_title='m (pixel)', yaxis_title='n (pixel)'))
    fig.add_trace(go.Heatmap(z=cube_topo, zmin=c_min, zmax=c_max, colorscale=color_topo_map, hovertemplate ='n : %{y}' + '<extra></extra>' + '<br> m : %{x}' + '<br> Value : %{z}', colorbar_title='Deflection'))
    fig.add_hline(y_select, line_dash="dash"); fig.add_vline(x_select, line_dash="dash")
    return fig

@st.cache_data(max_entries=1, show_spinner=False)
def topo_IRsection_x(cube, frequencies, x_select, color_IR_section, IR_max, IR_min, width_px, height_px) :
    fig2 = go.Figure(layout=dict(title='m='+str(x_select), height=height_px, width=width_px, xaxis_title='Frequency (kHz)', yaxis_title='n (pixel)'))
    fig2.add_trace(go.Heatmap(z=cube[:,x_select,:], x=frequencies, zmin=IR_min, zmax=IR_max, colorscale=color_IR_section, hovertemplate ='n : %{y}' + '<extra></extra>' + '<br> Frequency (kHz) : %{x}' + '<br> Value : %{z}', colorbar_title='Amplitude'))
    return fig2

@st.cache_data(max_entries=1, show_spinner=False)
def topo_IRsection_y(cube, frequencies, y_select, color_IR_section, c_max, c_min, width_px, height_px) :
    fig3 = go.Figure(layout=dict(title='n='+str(y_select), height=height_px, width=width_px, yaxis_title='Frequency (kHz)', xaxis_title='m (pixel)'))
    fig3.add_trace(go.Heatmap(z=cube[y_select,:,:].T, y=frequencies, zmin=IR_min, zmax=IR_max, colorscale=color_IR_section, hovertemplate ='m : %{x}' + '<br> Frequency (kHz)  : %{y}' + '<extra></extra>' +  '<br> Value : %{z}', colorbar_title='Amplitude'))
    return fig3

@st.cache_data(max_entries=1)
def multi_plot(frequency, cube, number, ymax, width, height, n_list, m_list, color_list):
    fig = go.Figure(layout=dict(title = 'Multiple data', width = width, height = height))
    for i in range(0, number):
        fig.add_scatter(x = frequency, y = cube[n_list[i], m_list[i]], mode='lines', line_color=color_list[i], name='n='+str(n_list[i])+'; m='+str(m_list[i]))
    fig.update_yaxes(title='Amplitude (mV)', range=[0, y_max])
    fig.update_xaxes(title='Frequency (kHz)', range=[min(frequency),max(frequency)])
    st.plotly_chart(fig, width='content')

@st.cache_data(max_entries=10, show_spinner=False)
def plot_results_pixels(cube, map_min, map_max, map_origin, color_map, colorbar_label, title, results_width, results_height, scale, bins_width, n, m, key, x=None, y=None, xy_unit=None):
    fig = make_subplots(rows=1, cols=2, column_widths=[4,1], subplot_titles = [title, colorbar_label])
    if map_origin=='upper' : yorder='reversed'
    elif map_origin=='lower' : yorder=True
    fig.add_trace(go.Heatmap(z=cube, zmin=map_min, zmax=map_max, colorscale=color_map, colorbar=dict(lenmode="pixels", len=results_height-70, x=0.7, thickness=results_height/25), hovertemplate ='n : %{y}' + '<extra></extra>' + '<br> m : %{x}' + '<br> Value : %{z}'), row=1, col=1)
    data_rescale = cube.reshape(n*m)[(cube.reshape(n*m)<=map_max) & (cube.reshape(n*m)>=map_min)]
    fig.add_trace(go.Histogram(y=data_rescale, ybins_size=bins_width, orientation='h', marker_color='grey', hovertemplate ='Range : %{y}' + '<extra></extra>' + '<br> Count : %{x}'), row=1, col=2)
    fig.update_layout(height=results_height, width=results_width, xaxis=dict(title='m (pixel)', scaleanchor="y"), yaxis_title='n (pixel)', xaxis2=dict(title='Count'), yaxis2 = dict(range=[map_min, map_max]), font_color='black', margin=dict(l=60, r=0, t=30, b=60, pad=0, autoexpand=False))
    fig.layout.annotations[1].update(x=0.81)
    fig.update_xaxes(ticks="outside", tickwidth=1, tickcolor='black', tickfont_color='black', title_font_color='black', ticklen=7)
    fig.update_yaxes(ticks="outside", autorange=yorder, tickwidth=1, tickcolor='black', tickfont_color='black', title_font_color='black', ticklen=7, col=1)
    fig.update_yaxes(showticklabels=False, col=2); fig.update_xaxes(col=2, tickangle=45, type=scale)
    return fig

def map_plots(cube, color_index, map_origin, colorbar_label, title, results_width, results_height, bins_width_initial, n, m, key, x, y, xy_unit, scale_units):
    with st.popover('Map and colorscale parameters', help="Enter the parameters of the map. Important : doing a zoom on the histogram won't change the limit of the colorbar."):
        color_map = st.selectbox('Colorscale for the map', st.session_state.colorscales, index=color_index, key=key+'_color')
        c1, c2 = st.columns(2)
        if np.nanmax(cube) > 1e5 : vmax = 1e5
        else : vmax = np.nanmax(cube)
        with c1 : map_max = st.number_input('Upper limit', value=vmax, key=key+'_max'); scale = st.radio('Count axis scale', ['linear', 'log'], horizontal=True, key=key+'_scale')
        with c2 : map_min = st.number_input('Lower limit', value=np.nanmin(cube), key=key+'_min'); bins_width = st.number_input('Width of intervals', min_value=0.001, value=float(bins_width_initial), key=key+'_bins')    
    if scale_units == None :
        with c1 : units = st.radio("Units to use:", ["Pixels", "Size"], key='units_'+key, horizontal=True)
        if units == 'Pixels' : fig = plot_results_pixels(cube, map_min, map_max, map_origin, color_map, colorbar_label, title, results_width, results_height, scale, bins_width, n, m, key, x=None, y=None, xy_unit=None)
        else : fig = plot_results_size(cube, map_min, map_max, map_origin, color_map, colorbar_label, title, results_width, results_height, scale, bins_width, n, m, key, x, y, xy_unit)
    elif scale_units == 'Pixels' : fig = plot_results_pixels(cube, map_min, map_max, map_origin, color_map, colorbar_label, title, results_width, results_height, scale, bins_width, n, m, key, x=None, y=None, xy_unit=None)
    elif scale_units == 'Size' : fig = plot_results_size(cube, map_min, map_max, map_origin, color_map, colorbar_label, title, results_width, results_height, scale, bins_width, n, m, key, x, y, xy_unit)
    return fig


@st.cache_data(max_entries=10, show_spinner=False)
def plot_results_size(cube, map_min, map_max, map_origin, color_map, colorbar_label, title, results_width, results_height, scale, bins_width, n, m, key, x, y, xy_unit):
    fig = make_subplots(rows=1, cols=2, column_widths=[4,1], subplot_titles = [title, colorbar_label])
    if map_origin=='upper' : yorder='reversed'
    elif map_origin=='lower' : yorder=True    
    fig.add_trace(go.Heatmap(z=cube, x=np.linspace(-x/2, x/2, m), y=np.linspace(-y/2, y/2, n), zmin=map_min, zmax=map_max, colorscale=color_map, colorbar=dict(lenmode="pixels", len=results_height-70, x=0.7, thickness=results_height/25), hovertemplate ='y : %{y}' + '<extra></extra>' + '<br> x : %{x}' + '<br> Value : %{z}'), row=1, col=1)
    data_rescale = cube.reshape(n*m)[(cube.reshape(n*m)<=map_max) & (cube.reshape(n*m)>=map_min)]
    fig.add_trace(go.Histogram(y=data_rescale, ybins_size=bins_width, orientation='h', marker_color='grey', hovertemplate ='Range : %{y}' + '<extra></extra>' + '<br> Count : %{x}'), row=1, col=2)
    fig.update_layout(height=results_height, width=results_width, xaxis=dict(title='x ('+xy_unit+')', scaleanchor="y"), yaxis_title='y ('+xy_unit+')', xaxis2=dict(title='Count'), yaxis2 = dict(range=[map_min, map_max]), font_color='black', margin=dict(l=60, r=0, t=30, b=60, pad=0, autoexpand=False))
    fig.layout.annotations[1].update(x=0.81)
    fig.update_xaxes(ticks="outside", tickwidth=1, tickcolor='black', tickfont_color='black', title_font_color='black', ticklen=7)
    fig.update_yaxes(ticks="outside", autorange=yorder, tickwidth=1, tickcolor='black', tickfont_color='black', title_font_color='black', ticklen=7, col=1)
    fig.update_yaxes(showticklabels=False, col=2); fig.update_xaxes(col=2, tickangle=45, type=scale)
    return fig

#%% Application
st.set_page_config(layout="wide")
st.title('Force Volume IR : cube processing')
configTab, visualisingTab, smoothingTab, peakrefTab, processingTab, integralTab, loadingTab, infoTab = st.tabs(["Configuration", "Visualising" ,"Smoothing", "Peak referencing", "Processing", "Integration by range", "Loading", ":information_source: Information"])

with configTab:
    with st.sidebar:
        with st.expander('See all the different colorscales'): st.write(px.colors.sequential.swatches_continuous()); st.write(px.colors.diverging.swatches_continuous())
    working_directory = Config()
    if working_directory == "" : st.error("Please, select a working directory") 
    else : 
        if [i for i in os.listdir(working_directory) if i.endswith(".spm")] == []:
            st.error("No .spm files found")
            st.stop()
        else :
            filenames = [i for i in os.listdir(working_directory) if i.endswith(".spm")]
            selected_filename = st.selectbox('Select a .spm file', filenames, on_change=reset_datas)
            filename = os.path.join(selected_filename).replace('.spm', '')
            try : st.session_state.n, st.session_state.m, st.session_state.l, st.session_state.x, st.session_state.y, st.session_state.xy_unit, chan_name, fwt, st.session_state.frequencies, st.session_state.cube, st.session_state.cube_topo = extraction(working_directory+filename)
            except MemoryError : st.error('Volume allocation exceed the available space. Try another file.'); st.session_state.FV_upload = False
            except RuntimeError : st.error('This file is not a Force-Volume file.'); st.session_state.FV_upload = False
            except OSError : st.error('The file cannot be read, maybe it is corrupted.')
            else : 
                st.write('You selected the .spm file : ', str(filename))
                st.write("File :", filename)
                st.write("Key parameters in data cube:")
                st.write("Pixels in XY =", str(st.session_state.n),"x",str(st.session_state.m))
                st.write("Size of the map = ",str(round(st.session_state.x,2)), 'x', str(round(st.session_state.y,2)),st.session_state.xy_unit)
                st.write("Channel = ", chan_name)
                st.write("Sweep parameter = ",fwt)
                st.write("The frequency range is from ",str(min(st.session_state.frequencies))," to ",str(max(st.session_state.frequencies))," with ",str(st.session_state.l)," points per spectrum.")
                st.session_state.FV_upload = True
            if 'Processed' not in os.listdir(working_directory) : os.mkdir(working_directory+'Processed'); st.success('"Processed" directory has been created in your working directory')
            else : st.info('"Processed" directory is already in your working directory')
            Saved_path=working_directory+"Processed/"
    
with visualisingTab:
    if st.session_state.FV_upload == True : 
        if st.button('Save topography as txt file.', type='primary') :
            np.savetxt(Saved_path+filename+"_TOPO.txt", st.session_state.cube_topo, delimiter=';')
            toast_appearance()
            st.toast('Topography has been saved', icon=':material/check:', duration="infinite")
        if st.toggle("Display IR sections along n and m axis") :
            with st.sidebar :
                with st.expander('Parameters of the 3 figures') :
                    c_n, c_m = st.columns(2)
                    with c_n : st.number_input('Enter a n value', min_value=0, max_value=st.session_state.n-1, key='y_select')
                    with c_m : st.number_input('Enter a m value', min_value=0, max_value=st.session_state.m-1, key='x_select')
                    color_topo_map = st.selectbox('Colorscale for the topography map', st.session_state.colorscales, index=103, key='color_topo_map')
                    c_l1, c_r1 = st.columns(2)
                    with c_l1 : c_max = st.number_input('Upper limit of the colorbar', value=np.max(st.session_state.cube_topo), key='c_max')
                    with c_r1 : c_min = st.number_input('Lower limit of the colorbar', value=np.min(st.session_state.cube_topo), key='c_min')
                    color_IR_section = st.selectbox('Colorscale for the IR sections map', st.session_state.colorscales, index=96, key='color_IR_section')
                    c_l2, c_r2 = st.columns(2)
                    with c_l2 : IR_max = st.number_input('Upper limit of the colorbar', value=np.max(st.session_state.cube[:,0,50:]), key='IR_max'); width_px = st.number_input('Width in pixels (for the 3 figure)', min_value=0, value=st.session_state.width_px, key='width_px')
                    with c_r2 : IR_min = st.number_input('Lower limit of the colorbar', value=0, key='IR_min'); height_px = st.number_input('Height in pixels (for the 3 figure)', min_value=0, value=st.session_state.height_px, key='height_px')
            c1, c2 = st.columns(2); c3, c4 = st.columns(2)
            with c1 :
                fig1 = topo_plot(st.session_state.cube_topo, st.session_state.color_topo_map, st.session_state.x_select, st.session_state.y_select, st.session_state.c_max, st.session_state.c_min, st.session_state.width_px, st.session_state.height_px)
                st.plotly_chart(fig1, False)
            with c2 :
                fig2 = topo_IRsection_x(st.session_state.cube, st.session_state.frequencies, st.session_state.x_select, st.session_state.color_IR_section, st.session_state.IR_max, st.session_state.IR_min, st.session_state.width_px, st.session_state.height_px)
                st.plotly_chart(fig2, False)
            
            with c3 :
                fig3 = topo_IRsection_y(st.session_state.cube, st.session_state.frequencies, st.session_state.y_select, st.session_state.color_IR_section, st.session_state.IR_max, st.session_state.IR_min, st.session_state.width_px, st.session_state.height_px)
                st.plotly_chart(fig3, False)
            with c4 :
                st.write('To download the IR sections datas along n and m, click on the buttons.')
                if st.button('Download .txt along m') :
                    np.savetxt(Saved_path+filename+"_sections_m"+str(st.session_state.y_select)+".txt", st.session_state.cube[:,st.session_state.x_select,:], fmt='%-10.8f', delimiter=';')
                    if filename+"_sections_m"+str(st.session_state.y_select)+".txt" in os.listdir(Saved_path):
                        st.write("Succesfully downloaded")
                if st.button('Download .txt along n') :
                    np.savetxt(Saved_path+filename+"_sections_n"+str(st.session_state.x_select)+".txt", st.session_state.cube[st.session_state.y_select,:,:], fmt='%-10.8f', delimiter=';')
                    if filename+"_sections_n"+str(st.session_state.x_select)+".txt"  in os.listdir(Saved_path):
                        st.write("Succesfully downloaded")
        else :
            fig_topo = map_plots(st.session_state.cube_topo, 103, 'lower', 'Deflection', 'Topography map', 810, 600, 500, st.session_state.n, st.session_state.m, 'TOPO', st.session_state.x, st.session_state.y, st.session_state.xy_unit, None)
            st.plotly_chart(fig_topo, width='content', on_select="ignore")
        if st.toggle("Display frequency spectrum", key='freq_plot') :
            container = st.container()
            with st.sidebar:
                with st.expander('Parameters of the frequency spectrum figures', expanded=True) :
                    choice = st.radio("Make a choice", ['Single plot', 'Multiple plot', 'Smoothing with Savitzky-Golay filter'])
                    c_1, c_2 = st.columns(2)
                    with c_1 : fig_width = st.number_input('Width of the figure in pixels', min_value=0, value=600, key='fig_width')
                    with c_2 : fig_height = st.number_input('Height of the figure in pixels', min_value=0, value=400, key='fig_height')
                    if choice == 'Single plot':
                        n_0 = st.number_input('Choose a n', min_value=0, max_value=st.session_state.n)
                        m_0 = st.number_input('Choose a m', min_value=0, max_value=st.session_state.m)
                        with container :
                            fig = px.line(x = st.session_state.frequencies, y = st.session_state.cube[n_0, m_0], labels=dict(x = "Frequency (kHz)", y = 'Amplitude (mV)'),
                                          title = 'Raw data', width = st.session_state.fig_width, height = st.session_state.fig_height,
                                          range_x = [min(st.session_state.frequencies),max(st.session_state.frequencies)], range_y = [np.min(st.session_state.cube[n_0, m_0]), np.max(st.session_state.cube[n_0, m_0,100:])])
                            st.plotly_chart(fig, width='content')
                    if choice == 'Multiple plot':
                        c_3, c_4 = st.columns(2)
                        with c_3 : number = st.number_input('Number of frequency spectrum to plot', min_value=2, max_value=10)
                        with c_4 : y_max = st.number_input('Change the amplitude axis max limit', min_value=0, value=15, key = 'y_max')
                        c7, c8, c9 = st.columns(3)
                        with c7: n_list = [st.number_input('n'+str(i), min_value=0, max_value=st.session_state.n-1, key = 'n_'+str(i)) for i in range(number)]
                        with c8 : m_list = [st.number_input('m'+str(i), min_value=0, max_value=st.session_state.m-1, key = 'm_'+str(i)) for i in range(number)]
                        with c9: color_list = [st.color_picker('Color picker', value = st.session_state.default_colors[i], key='color_'+str(i)) for i in range(number)]
                        with container : multi_plot(st.session_state.frequencies, st.session_state.cube, number, st.session_state.y_max, st.session_state.fig_width, st.session_state.fig_height, n_list, m_list, color_list)
                    if choice == 'Smoothing with Savitzky-Golay filter' :
                        c7, c8, c9 = st.columns(3)
                        with c7 : n_0 = st.number_input('n', min_value=0, max_value=st.session_state.n); window_length = st.number_input('Window length', min_value=1, value=11)
                        with c8 : m_0 = st.number_input('m', min_value=0, max_value=st.session_state.m); polyorder = st.number_input('Polynome order', min_value=0, value=3)
                        with c9: color_raw = st.color_picker('Color picker', value='#0035FD', key='color_raw'); color_smoothed = st.color_picker('Color picker', value='#FD0004', key='color_smoothed')
                        with container :
                            fig = px.line(x = st.session_state.frequencies, y = savgol_filter(st.session_state.cube[n_0, m_0], window_length,polyorder), labels=dict(x = "Frequency (kHz)", y = 'Amplitude (mV)', legend='Raw'), title = 'Raw vs smoothed data', width = st.session_state.fig_width, height = st.session_state.fig_height, range_x = [min(st.session_state.frequencies),max(st.session_state.frequencies)], range_y = [np.min(st.session_state.cube[n_0, m_0]), np.max(st.session_state.cube[n_0, m_0,100:])])
                            fig.add_scatter(x = st.session_state.frequencies, y = st.session_state.cube[n_0, m_0], mode='lines', zorder=10)
                            fig.update_traces(selector = 0, line_color=st.session_state.color_smoothed, name='Smoothed', showlegend=True);  fig.update_traces(selector = 1, line_color=st.session_state.color_raw, name='Raw')
                            fig.update_layout(template=None, font_color='black', yaxis_gridcolor='black', xaxis_gridcolor='black', yaxis_zerolinecolor='black')
                            fig.update_xaxes(ticks='outside', gridcolor='black', tickcolor='black', zeroline=True, zerolinecolor='black'); fig.update_yaxes(range=[0, np.nanmax(st.session_state.cube[n_0, m_0])], gridcolor='black', ticks='outside', tickcolor='black')
                            st.plotly_chart(fig, width='content')

with smoothingTab:  
    if st.session_state.FV_upload == True :
        c1_p, c2_p = st.columns([2,1])
        with c1_p:
            with st.container(border=True) :
                c1, c2 = st.columns([1,2])
                with c1 : st.write("Choose your Savitsky-Golay filter's parameters.") ; window_length = st.number_input('Window length', min_value=1, value=11); polyorder = st.number_input('Polynome order', min_value=0, value=3)
                with c2 :
                    st.write("Choose to apply an offset or not.")
                    offset_type = st.radio('Offset type', ['Range selection', 'Input personnalised value', 'None'], horizontal=True, key='offset_type', label_visibility='collapsed')
                    if offset_type=='Range selection' :
                        st.write('Select a frequency range (kHz) where the offset can be calculated, then applied to all spectra.')
                        c_fminsub, c_fmaxsub = st.columns(2)
                        freq_min_sub = c_fminsub.number_input('Start', min_value=st.session_state.frequencies[0], max_value=st.session_state.frequencies[-2], key='sub_f_min'); freq_max_sub = c_fmaxsub.number_input('End', min_value=st.session_state.frequencies[1], max_value=st.session_state.frequencies[-1], key='sub_f_max')
                        offset_values = [freq_min_sub, freq_max_sub]
                    elif offset_type=='Input personnalised value' : offset_values = st.number_input('Offset to remove (mV)', min_value=0.01, value=0.01, key='fixed_offset')
                    elif offset_type=='None' : st.write('No offset will be applied'); offset_values=0
                if st.button('Time to smooth'): st.session_state.cube_smoothed, st.session_state.carto_offset = smoothing_SG(st.session_state.cube, st.session_state.frequencies, window_length, polyorder, st.session_state.n, st.session_state.m, st.session_state.l, offset_type, offset_values)

        with c2_p :
            if st.button('Save current smoothing results.') :
                with st.spinner('Saving...'):
                    save_datas(Saved_path+filename+"_SMOOTHED.txt", st.session_state.cube_smoothed)
                    with open(Saved_path+filename+"_SMOOTHED.txt", 'r+') as file_txt:
                        file_data = file_txt.read(); file_txt.seek(0, 0)
                        if offset_type=='Range selection' : file_txt.write('# Subtract by mean between '+str(st.session_state.sub_f_min)+' to '+str(st.session_state.sub_f_max)+' kHz\n' + file_data) 
                        elif offset_type=='Input personnalised value' : file_txt.write('# Subtract a fixed offset of '+str(st.session_state.fixed_offset)+'mV\n' + file_data) 
                        elif offset_type=='None' : file_txt.write('# No offset subtraction\n' + file_data) 
                toast_appearance()
                st.toast('Smoothed datas has been saved !', duration="infinite")
            if st.button('Save integral of amplitude on all the frequency range.') :
                with st.spinner('Saving...'):
                    save_datas(Saved_path+filename+"_TOTAL-INTEGRAL.txt", np.trapz(st.session_state.cube_smoothed, st.session_state.frequencies))
                    with open(Saved_path+filename+"_TOTAL-INTEGRAL.txt", 'r+') as file_txt:
                        file_data = file_txt.read(); file_txt.seek(0, 0)
                        if offset_type=='Range selection' : file_txt.write('# Subtract by mean between '+str(st.session_state.sub_f_min)+' to '+str(st.session_state.sub_f_max)+' kHz\n' + file_data) 
                        elif offset_type=='Input personnalised value' : file_txt.write('# Subtract a fiwed offset of '+str(st.session_state.fixed_offset)+'mV\n' + file_data) 
                        elif offset_type=='None' : file_txt.write('# No offset subtraction\n' + file_data) 
                toast_appearance()
                st.toast('Smoothed datas has been saved !', duration="infinite")
            if st.button('Save value offset results.') :
                with st.spinner('Saving...'):
                    save_datas(Saved_path+filename+"_OFFSET.txt", st.session_state.carto_offset)
                    with open(Saved_path+filename+"_OFFSET.txt", 'r+') as file_txt:
                        file_data = file_txt.read(); file_txt.seek(0, 0)
                        if offset_type=='Range selection' : file_txt.write('# Subtract by mean between '+str(st.session_state.sub_f_min)+' to '+str(st.session_state.sub_f_max)+' kHz\n' + file_data) 
                        elif offset_type=='Input personnalised value' : file_txt.write('# Subtract a fiwed offset of '+str(st.session_state.fixed_offset)+'mV\n' + file_data) 
                        elif offset_type=='None' : file_txt.write('# No offset subtraction\n' + file_data) 
                toast_appearance()
                st.toast('Offset values has been saved !', duration="infinite")
            if filename+"_SMOOTHED.txt" in os.listdir(Saved_path):
                c1, c2 = st.columns([2,1], vertical_alignment='center')
                with c1 : st.info('Smoothed datas has been found in the "Processed directory", to load them click on the "Load" button.')
                with c2 :
                    if st.button('Load') :
                        SMOOTHED=Saved_path+filename+"_SMOOTHED.txt"
                        with st.spinner('Loading...'): cube_smoothed=np.loadtxt(SMOOTHED, dtype=np.float64).reshape((st.session_state.n,st.session_state.m,st.session_state.l)); st.session_state.cube_smoothed = cube_smoothed
                        with open(SMOOTHED, 'r') as file_txt: st.write(file_txt.readline()[1:])
        c1_rs, c2_rs = st.columns(2)
        with c1_rs :
            if 'cube_smoothed' in st.session_state:
                try : fig_smooth_IR = map_plots(np.trapz(st.session_state.cube_smoothed, st.session_state.frequencies), 38, 'lower', 'Integral (mV.kHz)', 'Integral on all the frequencies', 810, 600, 1000.00, st.session_state.n, st.session_state.m, 'SMOOTH', st.session_state.x, st.session_state.y, st.session_state.xy_unit, 'Pixels')
                except ValueError : pass
                else: st.plotly_chart(fig_smooth_IR, width='content')
        with c2_rs :
            if 'carto_offset' in st.session_state:
                try : fig_offset = map_plots(st.session_state.carto_offset, 0, 'lower', 'Offset (mV)', 'Subtracted offset', 810, 600, 0.01, st.session_state.n, st.session_state.m, 'OFFSET', st.session_state.x, st.session_state.y, st.session_state.xy_unit, 'Pixels')
                except ValueError : pass
                else: st.plotly_chart(fig_offset, width='content')

with peakrefTab:
    if st.session_state.FV_upload == True :
        c1, c2 = st.columns([3,1])
        with c1 : 
            with st.form("my-form", clear_on_submit=True):
                uploaded_file = st.file_uploader("To load a file with the peaks' parameters.")
                submitted = st.form_submit_button("UPLOAD!")
                if submitted and uploaded_file is not None:
                    st.session_state.df_Peaks_ref = pd.read_csv(uploaded_file, header=0, index_col=0)
                    old_name = st.session_state.df_Peaks_ref.columns
                    if len(old_name)==10 : new_name = ['center','freq_min','freq_max','y_lim', 'f0_shift', 'FWHM_shift', 'window_freq','FWHM0','FWHM_threshold','per_integrale']
                    else : new_name = ['center','freq_min','freq_max','y_lim', 'f0_shift', 'FWHM_shift', 'window_freq','FWHM_threshold','per_integrale']
                    st.session_state.df_Peaks_ref = st.session_state.df_Peaks_ref.rename(columns = {old_name[i]:new_name[i] for i in range(0,len(old_name))})
                    if 'FWHM0' in st.session_state.df_Peaks_ref.columns : st.session_state.df_Peaks_ref =st.session_state.df_Peaks_ref.drop('FWHM0', axis=1)
                    if 'FWHM_shift' not in st.session_state.df_Peaks_ref.columns : st.session_state.df_Peaks_ref = st.session_state.df_Peaks_ref.insert(5, 'FWHM_shift', 10)
        with c2 : 
            if filename+"_PeaksRef.csv" in os.listdir(Saved_path):
                st.info('A peaks reference file has been found in your "Processed" folder, click ont the Load button to load this file.')
                if st.button('Load', 'Peaks loading') :
                    st.session_state.df_Peaks_ref = pd.read_csv(Saved_path+filename+"_PeaksRef.csv", header=0, index_col=0)
                    old_name = st.session_state.df_Peaks_ref.columns
                    if len(old_name)==9 : new_name = ['center','freq_min','freq_max','y_lim', 'f0_shift', 'window_freq','FWHM0','FWHM_threshold','per_integrale']
                    else : new_name = ['center','freq_min','freq_max','y_lim', 'f0_shift', 'window_freq','FWHM_threshold','per_integrale']
                    st.session_state.df_Peaks_ref = st.session_state.df_Peaks_ref.rename(columns = {old_name[i]:new_name[i] for i in range(0,len(old_name))})
                    if 'FWHM0' in st.session_state.df_Peaks_ref.columns : st.session_state.df_Peaks_ref =st.session_state.df_Peaks_ref.drop('FWHM0', axis=1)
                    if 'FWHM_shift' not in st.session_state.df_Peaks_ref.columns : st.session_state.df_Peaks_ref.insert(5, 'FWHM_shift', 10)

        with st.form("Peak submit") :
            df_Peaks_ref = st.session_state.df_Peaks_ref
            df_Peaks_ref_edit = st.data_editor(df_Peaks_ref, num_rows="dynamic",
            column_config={"Peak n°": st.column_config.NumberColumn('Peak n°', min_value=1,step=1),
            'center' : st.column_config.NumberColumn('Center (kHz)', min_value=int(min(st.session_state.frequencies)), max_value=int(max(st.session_state.frequencies)), step=1, required=True),
            'freq_min' : st.column_config.NumberColumn('Minimal frequency (kHz)', min_value=min(st.session_state.frequencies), max_value=max(st.session_state.frequencies), required=True),
            'freq_max' : st.column_config.NumberColumn('Maximal frequency (kHz)', min_value=min(st.session_state.frequencies), max_value=max(st.session_state.frequencies), required=True),
            'y_lim' : st.column_config.NumberColumn('Amplitude threshold (mV)', min_value=0.00, max_value=np.max(st.session_state.cube), step=0.01, required=True, default=0.00, format='%.2f'),
            'f0_shift' : st.column_config.NumberColumn('Accepted shift of the central frequency (kHz)', required=True, min_value=0.05, step=0.01, default=2.),
            'FWHM_shift' : st.column_config.NumberColumn('Accepted shift of the FWHM (%)', required=True, min_value=0.05, step=0.01, default=10.),
            'window_freq' : st.column_config.NumberColumn('Window fit (kHz)', min_value=1.00, required=True, step=0.01, default=1.00),
            'FWHM_threshold' : st.column_config.NumberColumn('FWHM threshold', min_value=1.00, required=True, step=0.01, default=1.00),
            'per_integrale' : st.column_config.NumberColumn('FWHM ratio for integration window', min_value=0, required=True, default=3.)},
            hide_index=True,)
            if st.form_submit_button('Register'):
                st.session_state.df_Peaks_ref = df_Peaks_ref_edit
                st.info('Peaks are registered for this session.')
                st.session_state.nbr_peak = st.session_state.df_Peaks_ref.shape[0]
        c1_ref, c2_ref = st.columns([0.2,0.8])
        with c1_ref :
            with st.form('Fit plot') :
                c_data, c_fit = st.columns(2)
                c_data.radio('Data to use', ['Raw datas', 'Smoothed datas'], key='test_choice')
                c_fit.radio('Fit function to use', ['Asymetric SHO', 'SHO', 'None'], key='test_fit')
                choice = st.radio("Plot peak n°", st.session_state.df_Peaks_ref.index.values, horizontal=True)
                n_0 = st.number_input('Choose a n', min_value=0, max_value=st.session_state.n)
                m_0 = st.number_input('Choose a m', min_value=0, max_value=st.session_state.m)
                if st.form_submit_button('Plot') :
                    if st.session_state.test_choice == 'Raw datas' : st.session_state.cube_test = st.session_state.cube[n_0, m_0]
                    else : st.session_state.cube_test = st.session_state.cube_smoothed[n_0, m_0]
                    with c2_ref:
                        if st.session_state.test_fit=='SHO' :
                            if np.shape(st.session_state.df_Peaks_ref.index.values)[0] > 1: SHO_plot(st.session_state.frequencies, st.session_state.cube_test, st.session_state.test_choice, **dict(st.session_state.df_Peaks_ref.iloc[int(np.where(st.session_state.df_Peaks_ref.index==choice)[0])]))
                            else : SHO_plot(st.session_state.frequencies, st.session_state.cube_test, st.session_state.test_choice, **dict(st.session_state.df_Peaks_ref.iloc[0]))
                        elif st.session_state.test_fit=='Asymetric SHO' :
                            if np.shape(st.session_state.df_Peaks_ref.index.values)[0] > 1: SHO_asym_plot(st.session_state.frequencies, st.session_state.cube_test, st.session_state.test_choice, **dict(st.session_state.df_Peaks_ref.iloc[int(np.where(st.session_state.df_Peaks_ref.index==choice)[0])]))
                            else : SHO_asym_plot(st.session_state.frequencies, st.session_state.cube_test, st.session_state.test_choice, **dict(st.session_state.df_Peaks_ref.iloc[0]))
                        elif st.session_state.test_fit=='None':
                            if np.shape(st.session_state.df_Peaks_ref.index.values)[0] > 1: nofit_plot(st.session_state.frequencies, st.session_state.cube_test, st.session_state.test_choice, **dict(st.session_state.df_Peaks_ref.iloc[int(np.where(st.session_state.df_Peaks_ref.index==choice)[0])]))
                            else : nofit_plot(st.session_state.frequencies, st.session_state.cube_test, st.session_state.test_choice, **dict(st.session_state.df_Peaks_ref.iloc[0]))
        with c1_ref :
            if st.button(label="Save peaks' initial parameters as CSV"):
                df_Peaks_ref.to_csv(Saved_path+filename+"_PeaksRef.csv")
                st.success("The parameters for each peaks has been saved in the 'Processed' folder.")
    
with processingTab:
    if st.session_state.FV_upload == True :
        if 'params' not in st.session_state : st.session_state.params = []       
        with st.container(border=True):
            st.write("1. Select the data and peak to use")
            c1_proc, c2_proc, c3_proc, c4_proc = st.columns(4, vertical_alignment='top')
            data_choice = c1_proc.radio('Data to use', ['cube', 'cube_smoothed'], format_func= lambda x: {'cube':'Raw datas', 'cube_smoothed':'Smoothed Datas'}.get(x), horizontal=True)
            fit_choice = c2_proc.radio('Fit function to use', ['ASHO', 'SHO', 'None'], key='fit_choice', format_func=lambda x: {'ASHO': "Asymetric SHO",'SHO': "SHO"}.get(x), horizontal=True)
            choice_peak = c3_proc.number_input('Select a peak to fit', min_value=1, max_value=st.session_state.nbr_peak, step=1)
            if np.shape(st.session_state.df_Peaks_ref.index.values) != (1,): st.session_state.params = dict(st.session_state.df_Peaks_ref.iloc[int(np.where(st.session_state.df_Peaks_ref.index==choice_peak)[0])])
            else : st.session_state.params = dict(st.session_state.df_Peaks_ref.iloc[0])
            ncores = c4_proc.number_input('Number of cores to use (-1 = max)', min_value=-1, max_value=os.cpu_count(), step=1)
        col_left, col_right = st.columns(2, border = True)
        col_left.write("2. Process the selected peak with the selected fit function to use"); col_right.write("3. Save results of the peak")
        c5_proc, c6_proc = col_left.columns(2, vertical_alignment='top')
        c7_proc, c8_proc = col_right.columns(2, vertical_alignment='top')
        if c5_proc.button('Find parameters', type='primary') :
            st.session_state.cube_to_process = st.session_state[data_choice]
            st.session_state.datas_to_plot = []
            if st.session_state.fit_choice == 'SHO' :
                st.session_state.cube_Amp, st.session_state.cube_center,  st.session_state.cube_FWHM, st.session_state.cube_FWHM_datas, st.session_state.cube_Damping, st.session_state.cube_B0, st.session_state.cube_ymax, st.session_state.cube_R2 = SHO_parameters(st.session_state.frequencies, st.session_state.cube_to_process, st.session_state.n, st.session_state.m, st.session_state.params['freq_min'], st.session_state.params['freq_max'],st.session_state.params['y_lim'],st.session_state.params['f0_shift'],st.session_state.params['FWHM_shift'], st.session_state.params['window_freq'], ncores)
                flexible_success('Parameters of the SHO have been obtained for all (n, m) positions. The parameters are the following : maximal amplitude, B<sub>0</sub>, damping, FWHM and central frequency.')
            elif st.session_state.fit_choice == 'ASHO' :
                st.session_state.cube_Amp, st.session_state.cube_center, st.session_state.cube_x0, st.session_state.cube_FWHM, st.session_state.cube_FWHM_datas, st.session_state.cube_Damping, st.session_state.cube_B0, st.session_state.cube_f0, st.session_state.cube_ymax, st.session_state.cube_R2 = SHO_asym_parameters(st.session_state.frequencies, st.session_state.cube_to_process, st.session_state.n, st.session_state.m, st.session_state.params['freq_min'], st.session_state.params['freq_max'],st.session_state.params['y_lim'],st.session_state.params['f0_shift'],st.session_state.params['FWHM_shift'], st.session_state.params['window_freq'], ncores)
                flexible_success('Parameters of the asymetric SHO have been obtained for all (n, m) positions. The parameters are the following : maximal amplitude, B<sub>0</sub>, damping, FWHM, central frequency f<sub>0</sub>, x<sub>0</sub> and g<sub>0</sub.')
            elif st.session_state.fit_choice == 'None' :
                st.session_state.cube_FWHM_datas, st.session_state.cube_ymax, st.session_state.cube_center = no_fit(st.session_state.frequencies, st.session_state.cube_to_process, st.session_state.n, st.session_state.m, st.session_state.params['freq_min'], st.session_state.params['freq_max'], st.session_state.params['y_lim'])
                flexible_success('The obtained parameters are the following : maximal amplitude, FWHM and central frequency f<sub>0</sub>.')

        with c6_proc:
            if st.button('Compute integral', type='primary'):
                if st.session_state.fit_choice == 'SHO' : st.session_state.cube_Area_SHO, st.session_state.cube_Area_datas, st.session_state.integration_window = area_computing(st.session_state.frequencies, st.session_state.cube_to_process, st.session_state.n, st.session_state.m, st.session_state.cube_B0, st.session_state.cube_Damping, st.session_state.cube_FWHM, st.session_state.cube_center, st.session_state.params['FWHM_threshold'], st.session_state.params['per_integrale'])
                elif st.session_state.fit_choice == 'ASHO' : st.session_state.cube_Area_SHO, st.session_state.cube_Area_datas, st.session_state.integration_window = area_asym_computing(st.session_state.frequencies, st.session_state.cube_to_process, st.session_state.n, st.session_state.m, st.session_state.cube_B0, st.session_state.cube_Damping, st.session_state.cube_FWHM_datas, st.session_state.cube_f0, st.session_state.cube_x0, st.session_state.cube_center, st.session_state.params['FWHM_threshold'], st.session_state.params['per_integrale'])
                elif st.session_state.fit_choice == 'None' : st.session_state.cube_Area_datas, st.session_state.integration_window = area_computing_nofit(st.session_state.frequencies, st.session_state.cube_to_process, st.session_state.n, st.session_state.m, st.session_state.cube_FWHM_datas, st.session_state.cube_center, st.session_state.params['FWHM_threshold'], st.session_state.params['per_integrale'])
        
        with c7_proc.popover('Datas to save'):
            if st.session_state.fit_choice == 'None' : check_state = True; check_value = False
            else : check_state = False; check_value = True
            st.checkbox('SHO area', value=check_value, key='SHO_area_save', disabled=check_state); st.checkbox('SHO maximal amplitude', value=check_value, key='SHO_amp_save', disabled=check_state); st.checkbox('Datas area', value=check_state, key='datas_area_save'); st.checkbox('Datas maximal amplitude', value=check_state, key='datas_amp_save'); st.checkbox('SHO central frequency', value=check_value, key='central_freq_save', disabled=check_state); st.checkbox('SHO FWHM', value=check_value, key='SHO_FWHM_save', disabled=check_state); st.checkbox('Datas FWHM', value=check_state, key='datas_FWHM_save'); st.checkbox('SHO damping', value=False, key='SHO_damping_save', disabled=check_state); st.checkbox('Q factor', value=False, key='Q_factor_save'); st.checkbox('SHO B0 constant', value=False, key='SHO_B0_save', disabled=check_state); st.checkbox('R² fit score', value=False, key='R2_save', disabled=check_state)
            if st.session_state.fit_choice == 'ASHO' : st.checkbox('ASHO x0', value=False, key='SHO_x0_save'); st.checkbox('ASHO g0', value=False, key='SHO_f0_save')
               
        if c8_proc.button('Save results'):
            freq, FWHM_thresh = str(int(st.session_state.params['center'])), str(int(st.session_state.params['FWHM_threshold']))
            if st.session_state.SHO_area_save == True :
                np.savetxt(Saved_path+filename+"_AREA-"+fit_choice+"_"+freq+"kHz_FWHM-thresh_"+FWHM_thresh+".txt", st.session_state.cube_Area_SHO, delimiter=';')
                with open(Saved_path+filename+"_AREA-"+fit_choice+"_"+freq+"kHz_FWHM-thresh_"+FWHM_thresh+".txt", 'r+') as file_txt: 
                    file_data = file_txt.read()     
                    file_txt.seek(0, 0) 
                    file_txt.write('# Integration window : '+ str(st.session_state.integration_window) +'kHz\n' + file_data) 
            if st.session_state.datas_area_save == True :
                np.savetxt(Saved_path+filename+"_AREA-datas_"+freq+"kHz_FWHM-thresh_"+FWHM_thresh+".txt", st.session_state.cube_Area_datas, delimiter=';')
                with open(Saved_path+filename+"_AREA-datas_"+freq+"kHz_FWHM-thresh_"+FWHM_thresh+".txt", 'r+') as file_txt: 
                    file_data = file_txt.read()     
                    file_txt.seek(0, 0) 
                    file_txt.write('# Integration window : ' + str(st.session_state.integration_window) + 'kHz\n' + file_data) 
            if st.session_state.SHO_amp_save == True : np.savetxt(Saved_path+filename+"_AMP-"+fit_choice+"_"+freq+"kHz.txt", st.session_state.cube_Amp, delimiter=';')
            if st.session_state.datas_amp_save == True : np.savetxt(Saved_path+filename+"_AMP-datas_"+freq+"kHz.txt", st.session_state.cube_ymax, delimiter=';')
            if st.session_state.central_freq_save == True : np.savetxt(Saved_path+filename+"_F0-"+fit_choice+"_"+freq+"kHz.txt", st.session_state.cube_center, delimiter=';')
            if st.session_state.SHO_FWHM_save == True : np.savetxt(Saved_path+filename+"_FWHM-"+fit_choice+"_"+freq+"kHz.txt", st.session_state.cube_FWHM, delimiter=';')
            if st.session_state.datas_FWHM_save == True : np.savetxt(Saved_path+filename+"_FWHM-datas_"+freq+"kHz.txt", st.session_state.cube_FWHM_datas, delimiter=';')
            if st.session_state.SHO_damping_save == True : np.savetxt(Saved_path+filename+"_DAMPING-"+fit_choice+"_"+freq+"kHz.txt", st.session_state.cube_Damping, delimiter=';')
            if st.session_state.Q_factor_save == True : np.savetxt(Saved_path+filename+"_Q-factor-"+fit_choice+"_"+freq+"kHz.txt", st.session_state.cube_Q, delimiter=';')
            if st.session_state.SHO_B0_save == True : np.savetxt(Saved_path+filename+"_B0-"+fit_choice+"_"+freq+"kHz.txt", st.session_state.cube_B0, delimiter=';')
            if st.session_state.R2_save == True : np.savetxt(Saved_path+filename+"_R2-"+fit_choice+"_"+freq+"kHz.txt", st.session_state.cube_R2, delimiter=';')
            try :
                if st.session_state.SHO_x0_save == True : np.savetxt(Saved_path+filename+"_x0-ASHO_"+freq+"kHz.txt", st.session_state.cube_x0, delimiter=';')
                if st.session_state.SHO_f0_save == True : np.savetxt(Saved_path+filename+"_g0-ASHO_"+freq+"kHz.txt", st.session_state.cube_f0, delimiter=';')
            except AttributeError : pass
            st.success('Datas has been saved.')            
        
        st.divider()

        #### Results plot
        with st.expander('General parameters of the results figures') :
            c1_e, c2_e, c3_e, c4_e = st.columns(4, vertical_alignment='center')
            results_height = c1_e.number_input('Height in pixels', key='results_height', min_value=0, value=400); st.session_state.results_width = st.session_state.results_height*1.4
            c2_e.write('The width is {x} pixels.'.format(x=st.session_state.results_width))
            c3_e.radio('Origin of the map:', ['lower', 'upper'], key='map_origin', index=0, horizontal=True)
            c4_e.radio("Units to use:", ["Pixels", "Size"], key='map', horizontal=True, index=1)
        try : st.session_state.cube_Q = st.session_state.cube_center/st.session_state.cube_FWHM
        except ValueError : st.session_state.cube_Q = st.session_state.cube_center/st.session_state.cube_FWHM_datas 
        except TypeError : pass
        try : FWHM_thresh = st.session_state.params['FWHM_threshold'].values[0]
        except AttributeError : FWHM_thresh = st.session_state.params['FWHM_threshold']
        c1_res_3, c2_res_3, c3_res_3 = st.columns(3)
        if np.shape(st.session_state.cube_Area_SHO)!=(0,) and (st.session_state.fit_choice != 'None'):
            with c1_res_3 : ##### Left column
                ax1 = map_plots(st.session_state.cube_Area_SHO, 38, st.session_state.map_origin, "Area (mV.kHz)", 'a) Area SHO function' , st.session_state.results_width, st.session_state.results_height, 100, st.session_state.n, st.session_state.m, 'ax1', st.session_state.x, st.session_state.y, st.session_state.xy_unit, st.session_state.map)
                st.plotly_chart(ax1, width='content')
                ax4 = map_plots(st.session_state.cube_Area_datas, 38, st.session_state.map_origin, "Area (mV.kHz)", 'd) Area of datas' , st.session_state.results_width, st.session_state.results_height, 100, st.session_state.n, st.session_state.m, 'ax4', st.session_state.x, st.session_state.y, st.session_state.xy_unit, st.session_state.map)
                st.plotly_chart(ax4, width='content')
                ax7 = map_plots(st.session_state.cube_topo, 103, st.session_state.map_origin, 'Relative height (nm)', 'g) Topography' , st.session_state.results_width, st.session_state.results_height, 100, st.session_state.n, st.session_state.m, 'ax7', st.session_state.x, st.session_state.y, st.session_state.xy_unit, st.session_state.map)
                st.plotly_chart(ax7, width='content')
                ax10 = map_plots(st.session_state.cube_Q, 96, st.session_state.map_origin, 'Q factor', 'j) Q factor' , st.session_state.results_width, st.session_state.results_height, 100, st.session_state.n, st.session_state.m, 'ax10', st.session_state.x, st.session_state.y, st.session_state.xy_unit, st.session_state.map)
                st.plotly_chart(ax10, width='content')
            with c2_res_3 : ##### Middle column
                ax2 = map_plots(st.session_state.cube_Amp, 38, st.session_state.map_origin, 'Amplitude (mV)', 'b) Max amplitude' , st.session_state.results_width, st.session_state.results_height, 1, st.session_state.n, st.session_state.m, 'ax2', st.session_state.x, st.session_state.y, st.session_state.xy_unit, st.session_state.map)
                st.plotly_chart(ax2, width='content')
                ax5 = map_plots(st.session_state.cube_ymax, 38, st.session_state.map_origin, 'Amplitude (mV)', "e) Datas' maximal amplitude" , st.session_state.results_width, st.session_state.results_height, 1, st.session_state.n, st.session_state.m, 'ax5', st.session_state.x, st.session_state.y, st.session_state.xy_unit, st.session_state.map)
                st.plotly_chart(ax5, width='content')
                ax8 = map_plots(st.session_state.cube_center, 86, st.session_state.map_origin, 'Frequency (kHz)', 'h) Central frequency' , st.session_state.results_width, st.session_state.results_height, 1, st.session_state.n, st.session_state.m, 'ax8', st.session_state.x, st.session_state.y, st.session_state.xy_unit, st.session_state.map)
                st.plotly_chart(ax8, width='content')            
            with c3_res_3 : ##### Right column
                if st.session_state.fit_choice =='SHO' : ax3 = map_plots(st.session_state.cube_FWHM, 60, st.session_state.map_origin, 'FWHM (kHz)', 'c) FWHM SHO' , st.session_state.results_width, st.session_state.results_height, 5, st.session_state.n, st.session_state.m, 'ax3', st.session_state.x, st.session_state.y, st.session_state.xy_unit, st.session_state.map)
                elif st.session_state.fit_choice =='ASHO' : ax3 = map_plots(st.session_state.cube_Damping, 60, st.session_state.map_origin, 'Damping (kHz)', 'c) Damping ASHO' , st.session_state.results_width, st.session_state.results_height, 5, st.session_state.n, st.session_state.m, 'ax3', st.session_state.x, st.session_state.y, st.session_state.xy_unit, st.session_state.map)
                st.plotly_chart(ax3, width='content')
                ax6 = map_plots(st.session_state.cube_FWHM_datas, 60, st.session_state.map_origin, 'FWHM (kHz)', 'f) Datas FWHM' , st.session_state.results_width, st.session_state.results_height, 5, st.session_state.n, st.session_state.m, 'ax6', st.session_state.x, st.session_state.y, st.session_state.xy_unit, st.session_state.map)
                st.plotly_chart(ax6, width='content')
                ax9 = map_plots(st.session_state.cube_R2, 34, st.session_state.map_origin, 'R²', 'i) R² fit score' , st.session_state.results_width, st.session_state.results_height, 0.005, st.session_state.n, st.session_state.m, 'ax9', st.session_state.x, st.session_state.y, st.session_state.xy_unit, st.session_state.map)
                st.plotly_chart(ax9, width='content')
        elif (np.shape(st.session_state.cube_Area_datas)!=(0,)) and (st.session_state.fit_choice == 'None'):
            with c1_res_3 : ##### Left column
                ax1_notfit = map_plots(st.session_state.cube_Area_datas, 38, st.session_state.map_origin, "Area (mV.kHz)", 'a) Area of datas' , st.session_state.results_width, st.session_state.results_height, 100, st.session_state.n, st.session_state.m, 'ax1_notfit', st.session_state.x, st.session_state.y, st.session_state.xy_unit, st.session_state.map)
                st.plotly_chart(ax1_notfit, width='content')
                ax4_notfit = map_plots(st.session_state.cube_FWHM_datas, 60, st.session_state.map_origin, 'FWHM (kHz)', 'd) Datas FWHM' , st.session_state.results_width, st.session_state.results_height, 5, st.session_state.n, st.session_state.m, 'ax4_notfit', st.session_state.x, st.session_state.y, st.session_state.xy_unit, st.session_state.map)
                st.plotly_chart(ax4_notfit, width='content')
            with c2_res_3 : ##### Middle column
                ax2_notfit = map_plots(st.session_state.cube_ymax, 38, st.session_state.map_origin, 'Amplitude (mV)', "b) Datas' maximal amplitude" , st.session_state.results_width, st.session_state.results_height, 1, st.session_state.n, st.session_state.m, 'ax2_notfit', st.session_state.x, st.session_state.y, st.session_state.xy_unit, st.session_state.map)
                st.plotly_chart(ax2_notfit, width='content')
                ax5_notfit = map_plots(st.session_state.cube_center, 86, st.session_state.map_origin, 'Frequency (kHz)', 'e) Central frequency' , st.session_state.results_width, st.session_state.results_height, 1, st.session_state.n, st.session_state.m, 'ax5_notfit', st.session_state.x, st.session_state.y, st.session_state.xy_unit, st.session_state.map)
                st.plotly_chart(ax5_notfit, width='content')
            with c3_res_3 : ##### Right column
                ax3_notfit = map_plots(st.session_state.cube_topo, 103, st.session_state.map_origin, 'Relative height (nm)', 'c) Topography' , st.session_state.results_width, st.session_state.results_height, 100, st.session_state.n, st.session_state.m, 'ax3_notfit', st.session_state.x, st.session_state.y, st.session_state.xy_unit, st.session_state.map)
                st.plotly_chart(ax3_notfit, width='content')

with integralTab :
    if st.session_state.FV_upload == True :
        st.info('This tab is for people who want to integrate datas on a given frequency range. It will not give you central frequency, FWHM, or maximal amplitude.')
        datas_select, c_fmin, c_fmax, integ_calc, save_integ = st.columns(5, vertical_alignment='bottom')
        integrate_choice = datas_select.radio('Data to use', ['cube', 'cube_smoothed'], format_func= lambda x: {'cube':'Raw datas', 'cube_smoothed':'Smoothed Datas'}.get(x), horizontal=True, key='integrate_choice')
        integ_fmin = c_fmin.number_input('Start of the range (kHz)', min_value=st.session_state.frequencies[0], max_value=st.session_state.frequencies[-2])
        integ_fmax = c_fmax.number_input('End of the range (kHz)', min_value=st.session_state.frequencies[1] , max_value=st.session_state.frequencies[-1])
        if integ_calc.button('Calculate the integration'):
            condition = (st.session_state.frequencies>=integ_fmin)&(st.session_state.frequencies<=integ_fmax)
            st.session_state.range_integration = np.trapz(st.session_state[integrate_choice][:, :, condition], st.session_state.frequencies[condition])
            integration_result_map = map_plots(st.session_state.range_integration, 38, 'lower', 'Integral (mV.kHz)', f'Integral on the frequency range {integ_fmin} to {integ_fmax} kHz', 810, 600, 1000.00, st.session_state.n, st.session_state.m, 'INTEGRAL', st.session_state.x, st.session_state.y, st.session_state.xy_unit, 'Pixels')
            st.plotly_chart(integration_result_map, False)
        if save_integ.button('Save datas'): np.savetxt(Saved_path+filename+'_range-'+str(integ_fmin)+'-'+str(integ_fmax)+'kHz.txt', st.session_state.range_integration)

with loadingTab:
    if st.session_state.FV_upload == True :
        st.error('If FileNotFoundError : check the name of your file, if your file contains float number (ex : 300.0 instead of 300) it will not work. In this case, change directly the name of your file.')
        c1, c2 = st.columns([2.5,1])
        with c1 :
            Path_check = st.checkbox('Default path to research the datas are in the "Processed" folder. If your saved datas are in an other folder, check the box.', False)    
            if Path_check : LoadingPath = st.text_input('Enter the path here.')+'/'
            else : LoadingPath = Saved_path
            load_file_list = os.listdir(LoadingPath)
            for i in load_file_list:
                if (filename in i) and (('AREA' in i ) or ('AIRE' in i)) : st.write(i)
        with c2 :
            fit_function_load = st.radio('Fitting function', ['ASHO', 'SHO'], format_func=lambda x: {'ASHO': "Asymetric SHO",'SHO': "SHO"}.get(x), horizontal=True)
            c2_1, c2_2 = st.columns(2, vertical_alignment='bottom')
            with c2_1 :
                if 'freq' not in st.session_state : st.session_state.freq = 0
                freq = int(st.number_input('Frequency (kHz)'))
            with c2_2 :
                with st.popover('Datas to load'): st.session_state.load_datas = st.checkbox('SHO area', value=True, key='SHO_area_load'); st.checkbox('SHO maximal amplitude', value=True, key='SHO_amp_load'); st.checkbox('Datas area', value=False, key='datas_area_load'); st.checkbox('Datas maximal amplitude', value=False, key='datas_amp_load'); st.checkbox('SHO central frequency', value=True, key='central_freq_load'); st.checkbox('SHO FWHM', value=False, key='SHO_FWHM_load'); st.checkbox('SHO damping', value=True, key='SHO_damping_load'); st.checkbox('SHO B0 constant', value=False, key='SHO_B0_load'); st.checkbox('Asymetric SHO x0', value=False, key='SHO_x0_load'); st.checkbox('Asymetric SHO g0', value=False, key='SHO_g0_load'); st.checkbox('Fit score', value=False, key='SHO_R2_load')
            c2_3, c2_4 = st.columns(2, vertical_alignment='bottom')
            with c2_3 : FWHM_thresh = int(st.number_input('FWHM threshold (kHz)')); DT = int(st.number_input('If not FWHM threshold, enter damping threshold (kHz)'))
            with c2_4 :
                if st.button("Load datas"):
                    if st.session_state.SHO_area_load == True :
                        if filename+"_AIRE-"+str(fit_function_load)+"_"+str(freq)+"kHz_FWHM_thresh_"+str(FWHM_thresh)+".txt" in load_file_list : st.session_state.cube_Area_SHO_l=np.loadtxt(LoadingPath+filename+"_AIRE-"+str(fit_function_load)+"_"+str(freq)+"kHz_FWHM_thresh_"+str(FWHM_thresh)+".txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_Area_SHO_l"] = True
                        elif filename+"_AREA-"+str(fit_function_load)+"_"+str(freq)+"kHz_FWHM_thresh_"+str(FWHM_thresh)+".txt" in load_file_list : st.session_state.cube_Area_SHO_l=np.loadtxt(LoadingPath+filename+"_AREA-"+str(fit_function_load)+"_"+str(freq)+"kHz_FWHM_thresh_"+str(FWHM_thresh)+".txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_Area_SHO_l"] = True
                        elif filename+"_AIRE-"+str(fit_function_load)+"_"+str(freq)+"kHz_DThreshold_"+str(DT)+".txt" in load_file_list : st.session_state.cube_Area_SHO_l=np.loadtxt(LoadingPath+filename+"_AIRE-"+str(fit_function_load)+"_"+str(freq)+"kHz_DThreshold_"+str(DT)+".txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_Area_SHO_l"] = True
                        elif filename+"_AREA-"+str(fit_function_load)+"_"+str(freq)+"kHz_DThreshold_"+str(DT)+".txt" in load_file_list : st.session_state.cube_Area_SHO_l=np.loadtxt(LoadingPath+filename+"_AREA-"+str(fit_function_load)+"_"+str(freq)+"kHz_DThreshold_"+str(DT)+".txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_Area_SHO_l"] = True
                        else : toast_appearance() ; st.toast('SHO area file not found.', icon="🚨", duration="infinite")
                    if st.session_state.SHO_amp_load == True :
                        if filename+"_AMP-"+str(fit_function_load)+"_"+str(freq)+"kHz.txt" in load_file_list : st.session_state.cube_Amp_l=np.loadtxt(LoadingPath+filename+"_AMP-"+str(fit_function_load)+"_"+str(freq)+"kHz.txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_Amp_l"] = True
                        elif filename+"_AMP_"+str(freq)+"kHz.txt" in load_file_list : st.session_state.cube_Amp_l=np.loadtxt(LoadingPath+filename+"_AMP_"+str(freq)+"kHz.txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_Amp_l"] = True
                        else : toast_appearance() ; st.toast('SHO amplitude file not found.', icon="🚨", duration="infinite")
                    if st.session_state.datas_area_load == True :
                        if filename+"_AIRE-raw_"+str(freq)+"kHz_DThreshold_"+str(DT)+".txt" in load_file_list : st.session_state.cube_Area_datas_l=np.loadtxt(LoadingPath+filename+"_AIRE-raw_"+str(freq)+"kHz_DThreshold_"+str(DT)+".txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_Area_raw_l"] = True
                        elif filename+"_AREA-raw_"+str(freq)+"kHz_FWHM-thresh_"+str(FWHM_thresh)+".txt" in load_file_list : st.session_state.cube_Area_datas_l=np.loadtxt(LoadingPath+filename+"_AREA-raw_"+str(freq)+"kHz_FWHM-thresh_"+str(FWHM_thresh)+".txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_Area_raw_l"] = True
                        elif filename+"_AREA-datas_"+str(freq)+"kHz_FWHM-thresh_"+str(FWHM_thresh)+".txt" in load_file_list : st.session_state.cube_Area_datas_l=np.loadtxt(LoadingPath+filename+"_AREA-datas_"+str(freq)+"kHz_FWHM-thresh_"+str(FWHM_thresh)+".txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_Area_raw_l"] = True
                        elif filename+"_AREA-raw_"+str(freq)+"kHz_DThreshold_"+str(DT)+".txt" in load_file_list : st.session_state.cube_Area_datas_l=np.loadtxt(LoadingPath+filename+"_AREA-raw_"+str(freq)+"kHz_DThreshold_"+str(FWHM_thresh)+".txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_Area_SHO_l"] = True
                        else : toast_appearance() ; st.toast('Datas area file not found.', icon="🚨", duration="infinite")
                    if st.session_state.datas_amp_load == True :
                        if filename+"_AMP-raw_"+str(freq)+"kHz.txt" in load_file_list : st.session_state.cube_ymax_l=np.loadtxt(LoadingPath+filename+"_AMP-raw_"+str(freq)+"kHz.txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_ymax_l"] = True
                        elif filename+"_AMP-datas_"+str(freq)+"kHz.txt" in load_file_list : st.session_state.cube_ymax_l=np.loadtxt(LoadingPath+filename+"_AMP-datas_"+str(freq)+"kHz.txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_ymax_l"] = True
                        elif filename+"_datas-AMP_"+str(freq)+"kHz.txt" in load_file_list : st.session_state.cube_ymax_l=np.loadtxt(LoadingPath+filename+"_datas-AMP_"+str(freq)+"kHz.txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_ymax_l"] = True
                        else : toast_appearance() ; st.toast('Datas amplitude file not found.', icon="🚨", duration="infinite")
                    if st.session_state.central_freq_load == True :
                        if filename+"_F0-"+str(fit_function_load)+"_"+str(freq)+"kHz.txt" in load_file_list : st.session_state.cube_center_l=np.loadtxt(LoadingPath+filename+"_F0-"+str(fit_function_load)+"_"+str(freq)+"kHz.txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_center_l"] = True
                        elif filename+"_F0_"+str(freq)+"kHz.txt" in load_file_list : st.session_state.cube_center_l=np.loadtxt(LoadingPath+filename+"_F0_"+str(freq)+"kHz.txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_center_l"] = True
                        else : toast_appearance() ; st.toast('SHO central frequency file not found.', icon="🚨", duration="infinite")
                    if st.session_state.SHO_FWHM_load == True :
                        if filename+"_FWHM-"+str(fit_function_load)+"_"+str(freq)+"kHz.txt" in load_file_list : st.session_state.cube_FWHM_l=np.loadtxt(LoadingPath+filename+"_FWHM-"+str(fit_function_load)+"_"+str(freq)+"kHz.txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_FWHM_l"] = True
                        elif filename+"_FWMH_"+str(freq)+"kHz.txt" in load_file_list : st.session_state.cube_FWHM_l=np.loadtxt(LoadingPath+filename+"_FWMH_"+str(freq)+"kHz.txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_FWHM_l"] = True
                        else : toast_appearance() ; st.toast('FWHM damping file not found.', icon="🚨", duration="infinite")
                    if st.session_state.SHO_damping_load == True :
                        if filename+"_DAMPING-"+str(fit_function_load)+"_"+str(freq)+"kHz.txt" in load_file_list : st.session_state.cube_Damping_l=np.loadtxt(LoadingPath+filename+"_DAMPING-"+str(fit_function_load)+"_"+str(freq)+"kHz.txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_Damping_l"] = True
                        elif filename+"_DAMPING_"+str(freq)+"kHz.txt" in load_file_list : st.session_state.cube_Damping_l=np.loadtxt(LoadingPath+filename+"_DAMPING_"+str(freq)+"kHz.txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_Damping_l"] = True
                        else : toast_appearance() ; st.toast('SHO damping file not found.', icon="🚨", duration="infinite")
                    if st.session_state.SHO_B0_load == True :
                        if filename+"_B0-"+str(fit_function_load)+"_"+str(freq)+"kHz.txt" in load_file_list : st.session_state.cube_B0_l=np.loadtxt(LoadingPath+filename+"_B0-"+str(fit_function_load)+"_"+str(freq)+"kHz.txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_B0_l"] = True
                        elif filename+"_B0_"+str(freq)+"kHz.txt" in load_file_list : st.session_state.cube_B0_l=np.loadtxt(LoadingPath+filename+"_B0_"+str(freq)+"kHz.txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_B0_l"] = True
                        else : toast_appearance() ; st.toast('SHO B0 file not found.', icon="🚨", duration="infinite")
                    if st.session_state.SHO_x0_load == True :
                        if filename+"_x0-ASHO_"+str(freq)+"kHz.txt" in load_file_list : st.session_state.cube_x0_l=np.loadtxt(LoadingPath+filename+"_x0-ASHO_"+str(freq)+"kHz.txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_x0_l"] = True
                        else : toast_appearance() ; st.toast('Asymetric SHO x0 file not found.', icon="🚨", duration="infinite")
                    if st.session_state.SHO_g0_load == True :
                        if filename+"_g0-ASHO_"+str(freq)+"kHz.txt" in load_file_list : st.session_state.cube_g0_l=np.loadtxt(LoadingPath+filename+"_g0-ASHO_"+str(freq)+"kHz.txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_g0_l"] = True
                        else : toast_appearance() ; st.toast('Asymetric SHO g0 file not found.', icon="🚨", duration="infinite")
                    if st.session_state.SHO_R2_load == True :
                        if filename+"_R2-"+str(fit_function_load)+"_"+str(freq)+"kHz.txt" in load_file_list : st.session_state.cube_R2_l=np.loadtxt(LoadingPath+filename+"_R2-"+str(fit_function_load)+"_"+str(freq)+"kHz.txt", dtype=np.float64, delimiter=';').reshape((st.session_state.n,st.session_state.m)); st.session_state.loaded_datas["cube_R2_l"] = True
                        else : toast_appearance() ; st.toast('R² score fit file not found.', icon="🚨", duration="infinite")
                    st.session_state.cube_topo_l=st.session_state.cube_topo.copy()    
        load_to_plot=[i for i in st.session_state.loaded_datas if st.session_state.loaded_datas[i]==True]
        with st.expander('General parameters of the results figures') :
            c1_e, c2_e, c3_e, c4_e, c5_e = st.columns([0.15,0.25,0.15,0.15,0.15], vertical_alignment='center')
            with c1_e: results_height = st.number_input('Height in pixels', key='results_height_load', min_value=0, value=400)
            with c2_e: st.session_state.results_width_load = st.session_state.results_height_load*1.35; st.write('The width is {x} pixels.'.format(x=st.session_state.results_width))
            with c3_e: st.radio('Origin of the map', ['lower', 'upper'], key='map_origin_load', index=0, horizontal=True)
            with c4_e: st.radio("Units to use:", ["Pixels", "Size"], key='map_load', horizontal=True)
            with c5_e: st.number_input("Number of plots to show per row.", min_value=1, value=3, key='plots_per_row')

        if sum(st.session_state.loaded_datas.values())>=2 :
            if ('cube_center_l' in st.session_state) and ('cube_Damping_l' in st.session_state) : st.session_state.cube_Q=st.session_state.cube_center_l/(2*st.session_state.cube_Damping_l)
            if 'cube_Area_SHO_l' in st.session_state : st.session_state.plot_params_cube_Area_SHO_l = {'cube':st.session_state.cube_Area_SHO_l, 'color_index':38, 'map_origin':st.session_state.map_origin_load, 'colorbar_label':"Area (mV.kHz)", 'title':'Area SHO function', 'results_width':st.session_state.results_width_load, 'results_height':st.session_state.results_height_load, 'bins_width_initial':100, 'n':st.session_state.n, 'm':st.session_state.m, 'key':'ax1_l', 'x':st.session_state.x, 'y':st.session_state.y, 'xy_unit':st.session_state.xy_unit, 'scale_units':st.session_state.map_load}
            if 'cube_Amp_l' in st.session_state : st.session_state.plot_params_cube_Amp_l = {'cube':st.session_state.cube_Amp_l, 'color_index':38, 'map_origin':st.session_state.map_origin_load, 'colorbar_label':"Amplitude (mV)", 'title':'Max amplitude SHO function', 'results_width':st.session_state.results_width_load, 'results_height':st.session_state.results_height_load,  'bins_width_initial':5, 'n':st.session_state.n, 'm':st.session_state.m, 'key':'ax2_l', 'x':st.session_state.x, 'y':st.session_state.y, 'xy_unit':st.session_state.xy_unit, 'scale_units':st.session_state.map_load}
            if 'cube_Area_datas_l' in st.session_state : st.session_state.plot_params_cube_Area_datas_l = {'cube':st.session_state.cube_Area_datas_l, 'color_index':38, 'map_origin':st.session_state.map_origin_load, 'colorbar_label':"Area (mV.kHz)", 'title':"Datas' area", 'results_width':st.session_state.results_width_load, 'results_height':st.session_state.results_height_load,  'bins_width_initial':100, 'n':st.session_state.n, 'm':st.session_state.m, 'key':'ax1_3', 'x':st.session_state.x, 'y':st.session_state.y, 'xy_unit':st.session_state.xy_unit, 'scale_units':st.session_state.map_load}
            if 'cube_ymax_l' in st.session_state : st.session_state.plot_params_cube_Amp_l = {'cube':st.session_state.cube_ymax_l, 'color_index':38, 'map_origin':st.session_state.map_origin_load, 'colorbar_label':"Amplitude (mV)", 'title':"Datas' max amplitude", 'results_width':st.session_state.results_width_load, 'results_height':st.session_state.results_height_load,  'bins_width_initial':5, 'n':st.session_state.n, 'm':st.session_state.m, 'key':'ax4_l', 'x':st.session_state.x, 'y':st.session_state.y, 'xy_unit':st.session_state.xy_unit, 'scale_units':st.session_state.map_load}
            if 'cube_topo_l' in st.session_state : st.session_state.plot_params_cube_topo_l = {'cube':st.session_state.cube_topo, 'color_index':103, 'map_origin':st.session_state.map_origin_load, 'colorbar_label':"Height (nm)", 'title':"Topography", 'results_width':st.session_state.results_width_load, 'results_height':st.session_state.results_height_load,  'bins_width_initial':500, 'n':st.session_state.n, 'm':st.session_state.m, 'key':'ax5_l', 'x':st.session_state.x, 'y':st.session_state.y, 'xy_unit':st.session_state.xy_unit, 'scale_units':st.session_state.map_load}
            if 'cube_center_l' in st.session_state : st.session_state.plot_params_cube_center_l = {'cube':st.session_state.cube_center_l, 'color_index':86, 'map_origin':st.session_state.map_origin_load, 'colorbar_label':"Frequency (kHz)", 'title':"Central frequency SHO function", 'results_width':st.session_state.results_width_load, 'results_height':st.session_state.results_height_load,  'bins_width_initial':1, 'n':st.session_state.n, 'm':st.session_state.m, 'key':'ax6_l', 'x':st.session_state.x, 'y':st.session_state.y, 'xy_unit':st.session_state.xy_unit, 'scale_units':st.session_state.map_load}
            if 'cube_FWHM_l' in st.session_state : st.session_state.plot_params_cube_FWHM_l = {'cube':st.session_state.cube_FWHM_l, 'color_index':60, 'map_origin':st.session_state.map_origin_load, 'colorbar_label':"Frequency (kHz)", 'title':"FWHM SHO function", 'results_width':st.session_state.results_width_load, 'results_height':st.session_state.results_height_load,  'bins_width_initial':1, 'n':st.session_state.n, 'm':st.session_state.m, 'key':'ax7_l', 'x':st.session_state.x, 'y':st.session_state.y, 'xy_unit':st.session_state.xy_unit, 'scale_units':st.session_state.map_load}
            if 'cube_Damping_l' in st.session_state : st.session_state.plot_params_cube_Damping_l = {'cube':st.session_state.cube_Damping_l, 'color_index':60, 'map_origin':st.session_state.map_origin_load, 'colorbar_label':"Frequency (kHz)", 'title':"Damping SHO function", 'results_width':st.session_state.results_width_load, 'results_height':st.session_state.results_height_load,  'bins_width_initial':1, 'n':st.session_state.n, 'm':st.session_state.m, 'key':'ax8_l', 'x':st.session_state.x, 'y':st.session_state.y, 'xy_unit':st.session_state.xy_unit, 'scale_units':st.session_state.map_load}
            if 'cube_B0_l' in st.session_state : st.session_state.plot_params_cube_B0_l = {'cube':st.session_state.cube_B0_l, 'color_index':10, 'map_origin':st.session_state.map_origin_load, 'colorbar_label':"B0", 'title':"B0 SHO function", 'results_width':st.session_state.results_width_load, 'results_height':st.session_state.results_height_load,  'bins_width_initial':1, 'n':st.session_state.n, 'm':st.session_state.m, 'key':'ax9_l', 'x':st.session_state.x, 'y':st.session_state.y, 'xy_unit':st.session_state.xy_unit, 'scale_units':st.session_state.map_load}
            if 'cube_x0_l' in st.session_state : st.session_state.plot_params_cube_x0_l = {'cube':st.session_state.cube_x0_l, 'color_index':10, 'map_origin':st.session_state.map_origin_load, 'colorbar_label':"x0", 'title':"x0 asymetric SHO function", 'results_width':st.session_state.results_width_load, 'results_height':st.session_state.results_height_load,  'bins_width_initial':1, 'n':st.session_state.n, 'm':st.session_state.m, 'key':'ax10_l', 'x':st.session_state.x, 'y':st.session_state.y, 'xy_unit':st.session_state.xy_unit, 'scale_units':st.session_state.map_load}
            if 'cube_g0_l' in st.session_state : st.session_state.plot_params_cube_g0_l = {'cube':st.session_state.cube_g0_l, 'color_index':10, 'map_origin':st.session_state.map_origin_load, 'colorbar_label':"g0", 'title':"g0 asymetric SHO function", 'results_width':st.session_state.results_width_load, 'results_height':st.session_state.results_height_load,  'bins_width_initial':1, 'n':st.session_state.n, 'm':st.session_state.m, 'key':'ax11_l', 'x':st.session_state.x, 'y':st.session_state.y, 'xy_unit':st.session_state.xy_unit, 'scale_units':st.session_state.map_load}            
            if 'cube_Q_l' in st.session_state : st.session_state.plot_params_cube_Q_l = {'cube':st.session_state.cube_Q_l, 'color_index':10, 'map_origin':st.session_state.map_origin_load, 'colorbar_label':"Q factor", 'title':"Q factor", 'results_width':st.session_state.results_width_load, 'results_height':st.session_state.results_height_load,  'bins_width_initial':1, 'n':st.session_state.n, 'm':st.session_state.m, 'key':'ax12_l', 'x':st.session_state.x, 'y':st.session_state.y, 'xy_unit':st.session_state.xy_unit, 'scale_units':st.session_state.map_load}
            if 'cube_R2_l' in st.session_state : st.session_state.plot_params_cube_R2_l = {'cube':st.session_state.cube_R2_l, 'color_index':34, 'map_origin':st.session_state.map_origin_load, 'colorbar_label':"R²", 'title':"R² fit score", 'results_width':st.session_state.results_width_load, 'results_height':st.session_state.results_height_load,  'bins_width_initial':0.005, 'n':st.session_state.n, 'm':st.session_state.m, 'key':'ax12_l', 'x':st.session_state.x, 'y':st.session_state.y, 'xy_unit':st.session_state.xy_unit, 'scale_units':st.session_state.map_load}
            k=iter(load_to_plot)
            ls = st.columns(st.session_state.plots_per_row)
            for i in itertools.cycle(ls) :
                with i :
                    try:
                        k_to_plot = next(k)
                        st.session_state['ax_'+k_to_plot] = map_plots(**dict(st.session_state['plot_params_'+k_to_plot]))
                        st.plotly_chart(st.session_state['ax_'+k_to_plot], False)
                    except StopIteration: break
            
with infoTab :
    "This application is made to process the data cube from the Force-Volume AFM-IR (info of this mode here https://pubs.acs.org/doi/10.1021/acsnano.5c04015)."
    with st.expander("The application is organized with multiple Tab") :
        "- The sidebar will contains all the parameters for the plot, explore it carefully, some parameters will appear depending of the which plot and map you decide to display."
        "- Configuration tab : to extract IR data cube, different hard disk can be selected."
        "- Visualising tab : allow to explore the IR data cube according to position (n, m), frequency sweep can be plotted as 2D map, or as 2D plot (single, multiple plot, before vs. after Savitsky-Golay filter)."
        "- Smoothing tab : to do the smoothing by Savitsky-Golay filter on the entire cube, an offset is then substracted to the frequency sweep. The offset correspond to the mean of the amplitude on a frequency range (Start and End), it's process for each position. To find the best offset, it is highly required to plot multiple frequency sweep to find the good frequency range without frequency resonance on it (Visualising tab -> Display frequency spectrum ; Sidebar -> Parameters of the frequency spectrum figures -> Multiple plot)."
        st.markdown('''
        - Peak referencing tab : all resonance frequencies and their parameters must be listed in the table, then registered (Register button) before doing any fit. You can add as many row as needed (click on + on one of the last row's cell). The :blue[**last two parameters**] will have an influence only on the calculation of the area (Processing tab -> Compute the SHO and datas integral) 
            - Center (kHz) : will just be used for the name of the .txt file results.
            - Minimal frequency (kHz) & Maximal frequency (kHz) : frequency range to search for the maximal amplitude of the frequency resonance (Visualising tab -> Display IR sections along n and m axis to help fing the range).
            - Amplitude threshold (mV) : If the maximal amplitude of the frequency resonance at a position (n, m) is inferior to the threshold value, then it is considered as noise and no equation will be fitted at this position.
            - Accepted shift of the central frequency (kHz) : restrict the calculated central frequency resonance around the intial central frequency (determined by finding the max amplitude) as *f<sub>0</sub><sup>initial</sup> - f<sub>0</sub><sup>shift</sup><= f<sub>0</sub><sup>calc</sup> f<sub>0</sub><sup>initial</sup> + f<sub>0</sub><sup>shift</sup>*.
            - Window fit (kHz) : the resonance frequency will always be in the center of the window fit, so the window fit as to be chosen accordingly to one of the widest resonance frequency. Attention, if the window is so wide that it can see other frequency resonance, then poor results may be obtained.
            - :blue[**FWHM threshold (kHz)**] : due to the mechanical inhomogenity of some smaple, or to some error (acquisition or processing), some damping value can be very high. As the width of the integration window depends of the highest damping value found among the damping result, if the max damping is too high then the calculated area of the datas (not the SHO) can take in account the frequency resonance around. The damping threshold value allow to restrict the research of the max damping to the value striclty inferior to the threshold.
            - :blue[**FWHM ratio for integration window**] : after finding the max FWHM, the integration window is determined by multiply it by n time the FWHM. \n
        After the parameters are registered you can test them with the window just below.''', unsafe_allow_html=True)
        "- Processing tab : the registered parameters will be used to do a fit on the choosen frequency resonance (*Select a peak to fit*) on all the positions. The processing is done in two step, first the fit is done to get all the parameters (*Do the fit*), then the area of the SHO and the data are calculated (*Compute the SHO and datas integral*). At the end, the 2D map will be  displayed. To select whiwh datas to save, click on the *Datas to save* expander. Then you can save the results."
        "- Loading tab : to load already processed datas of the selected file in the Configuration tab."
    
    with st.expander("Details on the available equation for the fit"):
        "Two equation are available : the SHO and the asymetric SHO. The latter is useful if some frequency resonance are asymetric."
        c_SHO, c_ASHO = st.columns(2)
        with c_SHO :
            with st.container(border=True):
                st.subheader('SHO')
                st.latex(r'''amp(\omega) = \frac{B_{0}}{\sqrt{\left(\omega^{2}-\omega_{0}^{2}\right)^{2}+\left(\Gamma\omega\right)^{2}}}''')
                c_r, c_l = st.columns(2)            
                with c_r : st.latex(r'''\omega = 2\pi f''')
                with c_l : st.latex(r'''FWHM = \frac{\sqrt{3}\Gamma}{2\pi}''')
                "With :"
                "- ${B_{0}}$ a constant"
                "- ${\omega}$ the angular frequency"
                "- ${\omega_{0}}$ a the central angular frequency"
                "- ${\Gamma}$ the damping"
                "- ${f}$ the frequency"
                "- ${FWHM}$ the Full Width at Half Maximum"    
        with c_ASHO :
            with st.container(border=True):
                st.subheader('Asymetric SHO')
                st.latex(r'''amp(x) = \frac{B_{0}}{\sqrt{\left((x-x_{0})^{2}-g_{0}^{2}\right)^{2}+\left(\Gamma(x-x_{0})\right)^{2}}}''')
                "With :"
                "- ${B_{0}}$ a constant"
                "- ${x}$ the frequency"
                "- ${x_{0}}$ the normalized central frequency"
                "- ${\Gamma}$ the damping"
                "- ${g_{0}}$ the central frequency"
            
                
