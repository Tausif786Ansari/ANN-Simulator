import streamlit as st
import torch as t
def session():
    #---------Session State app.py---------
    if 'MODEL' not in st.session_state:
        st.session_state.MODEL = None
    if 'DATA_OPT' not in st.session_state:
        st.session_state.DATA_OPT = None
    if 'DATASET' not in st.session_state:
        st.session_state.DATASET = None 
    if 'DESIGN_MODEL' not in st.session_state:
        st.session_state.DESIGN_MODEL = False
    if 'PLOT_DATA_NAV' not in st.session_state:
        st.session_state.PLOT_DATA_NAV = {}
    if "SIDEBAR_FIG" not in st.session_state:
        st.session_state.SIDEBAR_FIG = None
    if "SHOW_SIDEBAR_PLOT" not in st.session_state:
        st.session_state.SHOW_SIDEBAR_PLOT = False
    #-------------Session State create data.py-------------
    if 'CREATE_DATA' not in st.session_state:
        st.session_state.CREATE_DATA = {}
    if 'DATA_TYPE' not in st.session_state:
        st.session_state.DATA_TYPE = None
    if 'DATA_DESIGN' not in st.session_state:
        st.session_state.DATA_DESIGN = None
    if 'PREVIEW_DATA' not in st.session_state:
        st.session_state.PREVIEW_DATA = False
    if 'DATA_CREATED' not in st.session_state:
        st.session_state.DATA_CREATED = False 
    if 'DATAFRAME' not in st.session_state:
        st.session_state.DATAFRAME = None
    #---------------Session State NeuralNetwork-------------
    if 'DATA_NORMAL' not in st.session_state:
        st.session_state.DATA_NORMAL = None
    if 'CREATE_MODEL' not in st.session_state:
        st.session_state.CREATE_MODEL = {}
    if "SIDE_MODEL_FIG" not in st.session_state:
        st.session_state.SIDE_MODEL_FIG = None
    if "SIDE_MODEL_PLOT" not in st.session_state:
        st.session_state.SIDE_MODEL_PLOT = False
    if 'SCALING' not in st.session_state:
        st.session_state.SCALING = None
    if 'SPLIT_RATIO' not in st.session_state:
        st.session_state.SPLIT_RATIO = 0.5
    if 'TRAIN_DEVICE' not in st.session_state:
        st.session_state.TRAIN_DEVICE = 'CPU'
    if 'START_TRAIN' not in st.session_state:
        st.session_state.START_TRAIN = False
    if 'TRAIN_OPTIONS' not in st.session_state:
        st.session_state.TRAIN_OPTIONS = None
    #--------------Training Model--------------
    if 'EPOCH' not in st.session_state:
        st.session_state.EPOCH = 0
    if 'TRAIN_SWITH' not in st.session_state:
        st.session_state.TRAIN_SWITH = False
    if 'TRAIN_LOSS' not in st.session_state:
        st.session_state.TRAIN_LOSS = None
    if 'TEST_LOSS' not in st.session_state:
        st.session_state.TEST_LOSS = None
    if 'CV_LOSS' not in st.session_state:
        st.session_state.CV_LOSS = None
    if 'TRAIN_LOSS_PLOT' not in st.session_state:
        st.session_state.TRAIN_LOSS_PLOT = []
    if 'TEST_LOSS_PLOT' not in st.session_state:
        st.session_state.TEST_LOSS_PLOT = []
    if 'CV_LOSS_PLOT' not in st.session_state:
        st.session_state.CV_LOSS_PLOT = []
    if 'MY_MODEL' not in st.session_state:
        st.session_state.MY_MODEL = None
    if 'DPLOT' not in st.session_state:
        st.session_state.DPLOT = None
    if 'LPLOT' not in st.session_state:
        st.session_state.LPLOT = None
    