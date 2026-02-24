import sessions as ss
import streamlit as st
import torch as t
import create_data as cd
import base64
import NeuralNetwork as NN
st.set_page_config("Training DL Model",layout="wide",initial_sidebar_state='expanded')
ss.session()
#------------Removing the top strip-------------------
st.markdown("""
        <style>

        /* Hide Streamlit header */
        header {visibility: hidden;}

        /* Hide top toolbar */
        div[data-testid="stToolbar"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
        }

        /* Hide footer */
        footer {visibility: hidden;}

        /* Remove top padding space */
        .block-container {
            padding-top: 1rem;
        }

        </style>
        """, unsafe_allow_html=True)

#---------------Function to change Background color of sidebar--------------
def sidebar_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
        <style>

        /* Sidebar base */
        [data-testid="stSidebar"] {{
            position: relative;
            background: transparent;
        }}

        /* Blurred image layer */
        [data-testid="stSidebar"]::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url("data:image/png;base64,{encoded}") center / cover no-repeat;
            filter: blur(6px);
            transform: scale(1.1);
            z-index: -1;
        }}

        /* Bring sidebar content forward */
        [data-testid="stSidebar"] > div {{
            position: relative;
            z-index: 1;
        }}

        </style>
        """, unsafe_allow_html=True)
    
#--------------Function to set background image blurr---------------
def set_blurry_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>

    /* Make app transparent */
    .stApp {{
        background: transparent;
    }}

    /* Background Image Layer */
    body::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: url("data:image/png;base64,{encoded}") center / cover no-repeat;
        filter: blur(16px);
        transform: scale(1.05);
        z-index: -2;
    }}

    /* Dark Overlay Layer */
    body::after {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(0,0,0,0.25);
        z-index: -1;
    }}

    /* Bring all Streamlit content to front */
    section.main, header, footer {{
        position: relative;
        z-index: 1;
    }}

    </style>
    """, unsafe_allow_html=True)

#----------------Function to chnage data in used Data---------------
def Change(dataset):
    st.session_state.DATASET = dataset
#---------------Background Image-----------------    
set_blurry_bg("D:/language/ML-DL projects/FirstModel/Project/Dataset Image/m2.jpg")  
#---------SideBar---------------
with st.sidebar:
    sidebar_bg("D:/language/ML-DL projects/FirstModel/Project/Dataset Image/m5.jpg")
    #-----------DL or ML-------------------------
    model=st.segmented_control("Technique",['Deep Learning','Machine Learning'],selection_mode="single",default=None)
    if model != st.session_state.MODEL:
        st.session_state.MODEL = model
        st.session_state.DATAFRAME = None
        st.session_state.DATA = [] 
        st.session_state.DATA_CREATED = False 
        st.session_state.START_TRAIN = False
        st.session_state.DESIGN_MODEL = False
        st.session_state.SIDE_MODEL_PLOT = False
        st.session_state.EPOCH = 0
    #-----------If DL model----------------------
    if st.session_state.MODEL:
        st.divider()
        #----------------Data Selection and Creation----------
        nav_data_opt=st.segmented_control("DATA",["Create Data","Use Data"],selection_mode="single",default=None)
        if nav_data_opt != st.session_state.DATA_OPT:
            st.session_state.DATA_OPT = nav_data_opt
            st.session_state.CREATE_DATA = {}
            st.session_state.DATA_CREATED = False
            st.session_state.PREVIEW_DATA = False
            st.session_state.DATA_DESIGN = None
            st.session_state.SHOW_SIDEBAR_PLOT = False
            st.session_state.START_TRAIN = False
            st.session_state.DESIGN_MODEL = False
            st.session_state.SIDE_MODEL_PLOT = False
            st.session_state.EPOCH = 0
            st.session_state.DPLOT = None
            st.session_state.TRAIN_LOSS_PLOT = []
            st.session_state.TEST_LOSS_PLOT = []
            st.session_state.CV_LOSS_PLOT = []
        #----------------Plotting on the sidebar-----------------------------
        if st.session_state.SHOW_SIDEBAR_PLOT and st.session_state.SIDEBAR_FIG:
            st.pyplot(st.session_state.SIDEBAR_FIG, use_container_width=True)
            
        if st.session_state.DATA_OPT == 'Use Data':
            #---------------Use Pre-stored Data--------------
            nav_data_set = None
            #-----------------row 1--------------------------
            col1, col2,col3 = st.columns([0.01,0.01,0.01],gap='xxsmall')
            with col1:
                st.image(r"D:\language\ML-DL projects\FirstModel\Project\Dataset Image\circle.png", width=62)
                st.button("Circle",width='content',on_click=Change,args=("Circle",))
            with col2:
                st.image(r"D:\language\ML-DL projects\FirstModel\Project\Dataset Image\swiss_roll.png", width=62)
                st.button("Swiss",width='content',on_click=Change,args=("Swiss",))
            with col3:
                st.image(r"D:\language\ML-DL projects\FirstModel\Project\Dataset Image\XOR.png", width=62)
                st.button("XOR D",width='content',on_click=Change,args=("XOR",))
            #-----------------row 2--------------------------
            col1, col2,col3 = st.columns([0.01,0.01,0.01],gap='xxsmall')
            with col1:
                st.image(r"D:\language\ML-DL projects\FirstModel\Project\Dataset Image\moon.png",width=62)
                st.button("Moon",width="content",on_click=Change,args=("Moon",))
            with col2:
                st.image(r"D:\language\ML-DL projects\FirstModel\Project\Dataset Image\Disc.png",width=62)
                st.button("Disc D",width="content",on_click=Change,args=("Disc",))
            with col3:
                st.image(r"D:\language\ML-DL projects\FirstModel\Project\Dataset Image\Binary.png",width=62)
                st.button("Linear",width="content",on_click=Change,args=("Binary",))
            #-----------------row 3 --------------------------
            col1, col2,col3 = st.columns([0.01,0.01,0.01],gap='xxsmall')
            with col1:
                st.image(r"D:\language\ML-DL projects\FirstModel\Project\Dataset Image\Spiral.png",width=62)
                st.button("Spiral",width="content",on_click=Change,args=("Spiral",))
            with col2:
                st.image(r"D:\language\ML-DL projects\FirstModel\Project\Dataset Image\Square.png",width=62)
                st.button("Square",width="content",on_click=Change,args=("Square",))
            with col3:
                st.image(r"D:\language\ML-DL projects\FirstModel\Project\Dataset Image\Board.png",width=62)
                st.button("Board",width="content",on_click=Change,args=("Board",))
            #------------------row 4--------------------------
            col1, col2,col3 = st.columns([0.01,0.01,0.01],gap='xxsmall')
            with col1:
                st.image(r"D:\language\ML-DL projects\FirstModel\Project\Dataset Image\Sin_D.png",width=62)
                st.button("Sine",width="content",on_click=Change,args=("Sine",))
            with col2:
                st.image(r"D:\language\ML-DL projects\FirstModel\Project\Dataset Image\Multi Swiss.png",width=62)
                st.button("swiss",width="content",on_click=Change,args=("Multi_Swiss",))
            with col3:
                st.image(r"D:\language\ML-DL projects\FirstModel\Project\Dataset Image\Ring.png",width=62)
                st.button("Ring",width="content",on_click=Change,args=('Ring',))
                    
            
        #----------------Ratio of Splitting Data (D-Train,D-Test)------------
        if st.session_state.DATAFRAME is not None:
            split_ratio = st.slider('Data Splitting Ratio',min_value=0.5,max_value=1.0,step=0.1)
            if split_ratio != st.session_state.SPLIT_RATIO:
                st.session_state.SPLIT_RATIO = split_ratio
            #-----------------Feature scaling technique--------------
            Scaling_Technique = st.selectbox("Select feature Scaling Technique",[None,'Standard Scaler','Min-Max Scaler','Robust Scaler','Mean Absolute Scaler','Unit Vector Normalization'],index=0)
            if Scaling_Technique != st.session_state.SCALING:
                st.session_state.SCALING = Scaling_Technique    
            
        #--------------Model Selection and creation-------------
        if st.session_state.MODEL == 'Deep Learning' and (st.session_state.DATAFRAME is not None):
            st.divider()
            col1,col2 = st.columns([0.01,0.01],gap='xxsmall')
            with col1:
                Design_model = st.segmented_control("Design ANN",['Design'],selection_mode='single',default=None,width='content')   
                if Design_model == 'Design':
                    st.session_state.DESIGN_MODEL = True
                else:
                    st.session_state.DESIGN_MODEL = False
                    st.session_state.START_TRAIN = False
            with col2:
                Train_device = st.segmented_control("Training Device",['CPU','GPU'],selection_mode='single',default=None,width='content')
                if Train_device != st.session_state.TRAIN_DEVICE:
                    st.session_state.TRAIN_DEVICE = Train_device
                    assert t.cuda.is_available(), st.warning('Cuda Not Available')
            if st.session_state.SIDE_MODEL_PLOT and st.session_state.SIDE_MODEL_FIG:
                st.pyplot(st.session_state.SIDE_MODEL_FIG,use_container_width=True)
        #----------------Model Training model--------------------
        if st.session_state.SIDE_MODEL_PLOT:
            st.divider()
            option = st.segmented_control("Train Model/Re-Design Model",["Start Training","Re-Design"],selection_mode='single',default=None,width='stretch')
            if st.session_state.TRAIN_OPTIONS != option:
                st.session_state.TRAIN_OPTIONS = option
                st.session_state.START_TRAIN = False
                st.session_state.TRAIN_LOSS_PLOT = []
                st.session_state.TEST_LOSS_PLOT = []
                st.session_state.CV_LOSS_PLOT = []
                st.session_state.MY_MODEL = None
                st.session_state.TRAIN_LOSS = 0
                st.session_state.TEST_LOSS = 0
                st.session_state.CV_LOSS = 0
                st.session_state.EPOCH = 0
                st.session_state.DPLOT = None
        if st.session_state.TRAIN_OPTIONS == 'Start Training':
            st.session_state.START_TRAIN = True 
        

#--------------Function for Home Page Content----------------
def home_content():
    # ===================== COMBINED ADVANCED CSS =====================
    st.markdown("""
        <style>
        /* ===== GLOBAL THEME & ANIMATIONS ===== */
        .stApp {
            background: radial-gradient(circle at top, #0f172a, #020617);
            animation: fadeIn 1.5s ease-in-out;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Neon Grid Overlay */
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(rgba(0, 255, 255, 0.05) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 255, 255, 0.05) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
            animation: gridPulse 4s ease-in-out infinite;
        }

        @keyframes gridPulse {
            0%, 100% { opacity: 0.05; }
            50% { opacity: 0.1; }
        }

        /* ===== SECTION CARD ===== */
        .section-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            border-radius: 25px;
            padding: 35px;
            margin: 35px 0;
            box-shadow: 
                0 0 50px rgba(0, 255, 255, 0.3),
                inset 0 0 20px rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(0, 255, 255, 0.2);
            transition: all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            position: relative;
            overflow: hidden;
        }

        .section-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.1), transparent);
            transition: left 0.6s;
        }

        .section-card:hover::before {
            left: 100%;
        }

        .section-card:hover {
            transform: scale(1.02) translateY(-5px);
            box-shadow: 
                0 0 80px rgba(0, 255, 255, 0.6),
                inset 0 0 30px rgba(255, 255, 255, 0.1);
        }

        /* ===== HEADINGS ===== */
        .title {
            font-size: 48px;
            font-weight: 900;
            background: linear-gradient(90deg, #00f2fe, #4facfe, #a855f7, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            text-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
            margin-bottom: 20px;
        }

        .subtitle {
            text-align: center;
            color: #cbd5f5;
            font-size: 20px;
            margin-bottom: 25px;
            font-weight: 500;
        }

        h3 {
            color: #38bdf8;
            margin-bottom: 20px;
            font-size: 28px;
            font-weight: 700;
            text-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
        }

        /* ===== TEXT ===== */
        .text {
            color: #e5e7eb;
            font-size: 17px;
            line-height: 1.8;
        }

        /* ===== CODE BLOCK ===== */
        pre {
            background: linear-gradient(135deg, #020617, #0f172a, #020617) !important;
            border-radius: 15px !important;
            border-left: 6px solid #38bdf8 !important;
            box-shadow: 0 0 30px rgba(56, 189, 248, 0.5);
            padding: 20px !important;
            margin: 20px 0 !important;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
        }

        /* ===== LIST CARD ===== */
        .list-card dt h5 {
            color: #7dd3fc;
            margin-top: 25px;
            font-size: 22px;
            font-weight: 600;
        }

        .list-card ul {
            list-style: none;
            padding-left: 0;
        }

        .list-card li {
            background: rgba(255, 255, 255, 0.08);
            margin: 12px 0;
            padding: 15px 18px 15px 45px;
            border-radius: 12px;
            position: relative;
            transition: all 0.4s ease;
            color: white;
            font-size: 16px;
        }

        .list-card li::before {
            content: "➤";
            position: absolute;
            left: 18px;
            color: #38bdf8;
            font-size: 18px;
        }

        .list-card li:hover {
            background: rgba(56, 189, 248, 0.3);
            transform: translateX(10px);
            box-shadow: 0 0 25px rgba(56, 189, 248, 0.7);
        }

        /* ===== GLASS TABLE ===== */
        .glass-wrap {
            display: flex;
            justify-content: center;
        }

        .glass-table {
            width: 100%;
            max-width: 800px;
            background: rgba(255, 255, 255, 0.12);
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 0 30px rgba(56, 189, 248, 0.4);
            border-collapse: collapse;
            backdrop-filter: blur(15px);
        }

        .glass-table th {
            background: linear-gradient(90deg, #38bdf8, #8b5cf6);
            color: black;
            padding: 16px;
            font-weight: 700;
            font-size: 16px;
        }

        .glass-table td {
            padding: 16px;
            color: white;
            text-align: center;
            font-size: 15px;
        }

        .glass-table tr:hover {
            background: rgba(56, 189, 248, 0.25);
            transition: background 0.3s ease;
        }

        /* ===== PIPELINE CARD ===== */
        .pipeline-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 22px;
            padding: 32px;
            margin-bottom: 30px;
            box-shadow: 
                0 0 30px rgba(0, 0, 0, 0.4),
                inset 0 0 15px rgba(0, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(0, 255, 255, 0.3);
            transition: all 0.4s ease;
            position: relative;
        }

        .pipeline-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 
                0 0 50px rgba(0, 255, 255, 0.7),
                inset 0 0 20px rgba(0, 255, 255, 0.2);
        }

        .pipeline-title {
            font-size: 30px;
            font-weight: 800;
            background: linear-gradient(90deg, #00f2fe, #4facfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 15px;
            text-shadow: 0 0 15px rgba(0, 255, 255, 0.6);
        }

        .pipeline-subtitle {
            font-size: 20px;
            color: #dbeafe;
            margin-top: 15px;
            font-weight: 600;
        }

        .pipeline-text {
            font-size: 17px;
            color: #f1f5f9;
            line-height: 1.8;
        }

        .pipeline-list {
            padding-left: 25px;
        }

        .pipeline-list li {
            color: #e0f2fe;
            margin: 8px 0;
            transition: all 0.3s ease;
            font-size: 16px;
        }

        .pipeline-list li:hover {
            color: #22d3ee;
            transform: translateX(8px);
        }

        .highlight {
            background: linear-gradient(90deg, #22d3ee, #0ea5e9);
            padding: 14px 20px;
            border-radius: 12px;
            color: black;
            font-weight: 700;
            margin-top: 18px;
            text-align: center;
            box-shadow: 0 0 15px rgba(34, 211, 238, 0.5);
        }

        /* ===== METAL CARD ===== */
        .metal-card {
            background: linear-gradient(145deg, #020617, #0f172a, #020617);
            border-radius: 24px;
            padding: 35px;
            margin: 40px 0;
            box-shadow: 
                inset 0 0 30px rgba(0, 255, 255, 0.2),
                0 0 40px rgba(0, 255, 255, 0.4);
            border: 1px solid rgba(0, 255, 255, 0.4);
            transition: all 0.5s ease;
            position: relative;
            overflow: hidden;
        }

        .metal-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.1), transparent);
            transition: left 0.7s;
        }

        .metal-card:hover::before {
            left: 100%;
        }

        .metal-card:hover {
            transform: scale(1.02);
            box-shadow: 
                inset 0 0 35px rgba(0, 255, 255, 0.5),
                0 0 70px rgba(0, 255, 255, 0.9);
        }

        .metal-title {
            font-size: 34px;
            font-weight: 900;
            background: linear-gradient(90deg, #00f2fe, #38bdf8, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 0 15px rgba(0, 255, 255, 0.6);
            margin-bottom: 20px;
        }

        .metal-text {
            color: #e5e7eb;
            font-size: 17px;
            line-height: 1.8;
        }

        .metal-list {
            list-style: none;
            padding-left: 0;
        }

        .metal-list li {
            background: rgba(255, 255, 255, 0.06);
            margin: 12px 0;
            padding: 14px 18px 14px 45px;
            border-radius: 12px;
            position: relative;
            color: #e0f2fe;
            transition: all 0.4s ease;
            font-size: 16px;
        }

        .metal-list li::before {
            content: "⛓";
            position: absolute;
            left: 18px;
            color: #22d3ee;
            font-size: 18px;
        }

        .metal-list li:hover {
            background: rgba(34, 211, 238, 0.25);
            transform: translateX(10px);
            box-shadow: 0 0 20px rgba(34, 211, 238, 0.8);
        }

        .metal-strip {
            background: linear-gradient(90deg, #22d3ee, #38bdf8, #a855f7);
            color: black;
            padding: 12px 20px;
            border-radius: 14px;
            font-weight: 700;
            display: inline-block;
            margin-top: 18px;
            box-shadow: 0 0 15px rgba(34, 211, 238, 0.5);
        }

        /* ===== RESPONSIVE DESIGN ===== */
        @media (max-width: 768px) {
            .title { font-size: 36px; }
            .section-card, .pipeline-card, .metal-card { padding: 20px; margin: 20px 0; }
            .glass-table { font-size: 14px; }
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("""
        <style>

        /* ===== CENTER CONTAINER ===== */
        .metal-title-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 40px;
        }

        /* ===== FLOATING METALLIC PLATFORM ===== */
        .metal-title-box {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            border-radius: 25px;
            padding: 35px;
            width:1200px;
            margin: 35px 0;
            box-shadow: 
                0 0 50px rgba(0, 255, 255, 0.3),
                inset 0 0 20px rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(0, 255, 255, 0.2);
            transition: all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            position: relative;
            overflow: hidden;
            
            /* Metallic border */
            border: 1px solid rgba(255,255,255,0.15);

            /* Floating effect */
            transform: perspective(1000px);
            
            /* Glow shadows */
            box-shadow:
                0 20px 40px rgba(0,0,0,0.6),
                0 0 30px rgba(0,255,255,0.4),
                inset 0 0 15px rgba(255,255,255,0.08);

            backdrop-filter: blur(10px);

            text-align: center;
            transition: 0.4s ease;
        }

        /* Hover floating animation */
        .metal-title-box:hover {
            transform: perspective(1000px) rotateX(0deg) translateY(-8px);
            box-shadow:
                0 30px 60px rgba(0,0,0,0.7),
                0 0 50px rgba(0,255,255,0.8),
                inset 0 0 20px rgba(255,255,255,0.12);
        }

        /* ===== TITLE TEXT ===== */
        .metal-title-text {
            font-size: 48px;
            font-weight: 900;
            background: linear-gradient(90deg, #00f2fe, #4facfe, #a855f7, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            text-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
            margin-bottom: 20px;
        }

        </style>
        """, unsafe_allow_html=True)


    st.markdown("""
        <div class="metal-title-wrapper">
            <div class="metal-title-box">
                <div class="metal-title-text">
                    Designing and Training AI Model
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    # ===================== HERO =====================
    st.markdown("""
        <div class="section-card">
        <div class="title">Machine Learning & Deep Learning</div>

        <div class="subtitle">
        A Complete End-to-End Training Pipeline — From Data to Deployment
        </div>

        <p class="text">
        Artificial Intelligence today is largely driven by Machine Learning (ML) and Deep Learning (DL).
        While both aim to learn patterns from data, they differ in scale, architecture, and complexity.
        </p>

        <p class="text">
        This blog will walk you through:
        </p>

        <ul class="text">
        <li>What ML and DL are</li>
        <li>Key differences between ML and DL</li>
        <li>The entire lifecycle of training ML & DL models</li>
        <li>Practical best practices at each stage</li>
        </ul>

        <p class="text">
        Think of this as a roadmap from raw data → trained model → real-world system.
        </p>
        </div>
        """, unsafe_allow_html=True)

    # ===================== ML =====================
    st.markdown("""
        <div class="section-card">
        <h3>1. What is Machine Learning (ML)?</h3>
        <p class="text">
        Machine Learning is a subset of AI where algorithms learn patterns from data and make predictions
        or decisions without explicit programming.
        </p>

        <p class="text">Instead of writing rules:</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
        <div class="section-card list-card">
        <dl>
        <dt><h5>Common ML Algorithm Families</h5></dt>
        <ul>
        <li>Linear Regression</li>
        <li>Logistic Regression</li>
        <li>KNN</li>
        <li>Decision Trees</li>
        <li>Random Forest</li>
        <li>SVM</li>
        <li>Naive Bayes</li>
        </ul>

        <dt><h5>Typical Use Cases</h5></dt>
        <ul>
        <li>Spam Detection</li>
        <li>Credit Scoring</li>
        <li>Sales Forecasting</li>
        <li>Recommendation Systems</li>
        </ul>
        </dl>
        </div>
        """, unsafe_allow_html=True)

    # ===================== DL =====================
    st.markdown("""
        <div class="section-card list-card">
        <h3>2. What is Deep Learning (DL)?</h3>
        <dl>
        <dt><h5>DL Excels When</h5></dt>
        <ul>
        <li>Data is very large</li>
        <li>Unstructured data (images, audio, text)</li>
        <li>Feature engineering is difficult</li>
        </ul>
        <dt><h5>Common Architectures</h5></dt>
        <ul>
        <li>ANN</li>
        <li>CNN</li>
        <li>RNN</li>
        <li>LSTM / GRU</li>
        <li>Transformers</li>
        </ul>
        <dt><h5>Typical Use Cases</h5></dt>
        <ul>
        <li>Image Recognition</li>
        <li>Speech-to-Text</li>
        <li>Chatbots / LLMs</li>
        <li>Autonomous Driving</li>
        </ul>
        </dl>
        </div>
    """, unsafe_allow_html=True)

    # ===================== TABLE =====================
    st.markdown("""
        <div class="section-card">
        <h3>3. ML vs DL (High-Level Comparison)</h3>
        <div class="glass-wrap">
        <table class="glass-table">
        <tr>
        <th>Aspect</th>
        <th>Machine Learning</th>
        <th>Deep Learning</th>
        </tr>
        <tr><td>Feature Engineering</td><td>Mostly Manual</td><td>Automatic</td></tr>
        <tr><td>Data Requirement</td><td>Small-Medium</td><td>Large</td></tr>
        <tr><td>Training Time</td><td>Short</td><td>Long</td></tr>
        <tr><td>Interpretability</td><td>Easier</td><td>Harder</td></tr>
        <tr><td>Hardware</td><td>CPU Enough</td><td>GPU / TPU Preferred</td></tr>
        </table>
        </div>
        </div>
    """, unsafe_allow_html=True)
    
    #---------------- OL------------------------
    st.markdown("""
        <div class="section-card list-card">
        <h3>End-to-End Training Pipeline (ML & DL)</h3>
        <dl>
        <dt><h5>The lifecycle can be divided into 8 major stages</h5></dt>
        <li>Problem Definition</li>
        <li>Data Collection</li>
        <li>Data Preprocessing</li>
        <li>Exploratory Data Analysis (EDA)</li>
        <li>Feature Engineering</li>
        <li>Model Building</li>
        <li>Model Training</li>
        <li>Evaluation</li>
        <li>Hyperparameter Tuning</li>
        <li>Post-Processing & Deployment</li>
        </ul>
        <dt>Let's explore each in depth.</dt>
        </dl>
        </div>
    """, unsafe_allow_html=True)

    # ---------------- CONTENT ----------------
    st.markdown("""
        <div class="pipeline-card">
        <div class="pipeline-title">1. Problem Definition</div>
        <p class="pipeline-text">
        Clarify what exactly you want the model to solve.
        </p>
        <div class="pipeline-subtitle">Examples</div>
        <ul class="pipeline-list">
        <li>Predict house price → Regression</li>
        <li>Spam or not → Classification</li>
        <li>Group customers → Clustering</li>
        </ul>
        <div class="pipeline-subtitle">Define Success Metric</div>
        <ul class="pipeline-list">
        <li>Accuracy</li>
        <li>F1-score</li>
        <li>RMSE</li>
        <li>AUC</li>
        </ul>
        <div class="highlight">
        Good problem definition = 50% of success.
        </div>
        </div>
        <div class="pipeline-card">
        <div class="pipeline-title">2. Data Collection</div>
        <ul class="pipeline-list">
        <li>CSV / Excel</li>
        <li>Databases</li>
        <li>APIs</li>
        <li>Sensors / Logs</li>
        <li>Web scraping</li>
        </ul>
        <div class="highlight">
        Model quality can never exceed data quality.
        </div>
        </div>
        <div class="pipeline-card">
        <div class="pipeline-title">3. Data Preprocessing</div>
        <div class="pipeline-subtitle">Handle Missing Values</div>
        <ul class="pipeline-list">
        <li>Drop rows</li>
        <li>Fill with mean / median / mode</li>
        <li>Predict using model</li>
        </ul>
        <div class="pipeline-subtitle">Handle Outliers</div>
        <ul class="pipeline-list">
        <li>IQR method</li>
        <li>Z-score</li>
        <li>Winsorization</li>
        </ul>
        <div class="pipeline-subtitle">Encoding Categorical Variables</div>
        <ul class="pipeline-list">
        <li>Label Encoding</li>
        <li>One-Hot Encoding</li>
        <li>Target Encoding</li>
        </ul>
        <div class="pipeline-subtitle">Scaling / Normalization</div>
        <ul class="pipeline-list">
        <li>Min-Max Scaling</li>
        <li>Standardization (Z-score)</li>
        </ul>
        <p class="pipeline-text">
        Gradient-based algorithms converge faster when features are on similar scales.
        </p>
        </div>
        <div class="pipeline-card">
        <div class="pipeline-title">4. Exploratory Data Analysis (EDA)</div>
        <ul class="pipeline-list">
        <li>Understand data distribution</li>
        <li>Detect anomalies</li>
        <li>Discover relationships</li>
        </ul>
        <div class="pipeline-subtitle">Common Techniques</div>
        <ul class="pipeline-list">
        <li>Histograms</li>
        <li>Box plots</li>
        <li>Correlation matrix</li>
        <li>Pair plots</li>
        </ul>
        </div>
        <div class="pipeline-card">
        <div class="pipeline-title">5. Feature Engineering</div>
        <div class="pipeline-subtitle">In Machine Learning</div>
        <ul class="pipeline-list">
        <li>Ratios</li>
        <li>Aggregations</li>
        <li>Polynomial terms</li>
        <li>Binning</li>
        </ul>
        <p class="pipeline-text">
        Example: BMI = weight / height²
        </p>
        <div class="pipeline-subtitle">In Deep Learning</div>
        <ul class="pipeline-list">
        <li>Tokenization for text</li>
        <li>Image resizing</li>
        <li>Spectrograms for audio</li>
        </ul>
        </div>
        <div class="pipeline-card">
        <div class="pipeline-title">6. Train – Test Split</div>
        <ul class="pipeline-list">
        <li>Training set</li>
        <li>Validation set</li>
        <li>Test set</li>
        </ul>
        <p class="pipeline-text">
        70% Train | 15% Validation | 15% Test
        </p>
        <div class="highlight">
        Train → Learn | Validation → Tune | Test → Final Evaluation
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- STEP 6 WHY ----------------
    st.markdown("""
        <div class="metal-card">
        <div class="metal-title">Why Split Data?</div>
        <ul class="metal-list">
        <li>Train → Learn</li>
        <li>Validation → Tune</li>
        <li>Test → Final Evaluation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- STEP 7 ----------------
    st.markdown("""
        <div class="metal-card">
        <div class="metal-title">7. Model Building</div>
        <p class="metal-text"><b>ML Example</b></p>
        </div>
        """, unsafe_allow_html=True)

    st.code("""
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        """, language="python")

    st.markdown("""
        <div class="metal-card">
        <p class="metal-text"><b>DL Example</b></p>
        </div>
        """, unsafe_allow_html=True)

    st.code("""
        model = Sequential([
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')])
        """, language="python")

    st.markdown("""
        <div class="metal-card">
        <ul class="metal-list">
        <li>Choose Architecture</li>
        <li>Number of Layers</li>
        <li>Activation Functions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- STEP 8 ----------------
    st.markdown("""
        <div class="metal-card">
        <div class="metal-title">8. Model Training</div>
        <p class="metal-text"><b>ML Training</b></p>
        </div>
        """, unsafe_allow_html=True)

    st.code("""model.fit(X_train, y_train)
        """, language="python")

    st.markdown("""
        <div class="metal-card">
        <p class="metal-text">The algorithm finds parameters that minimize loss.</p>
        <p class="metal-text"><b>DL Training Uses</b></p>
        <ul class="metal-list">
        <li>Forward Propagation</li>
        <li>Loss Computation</li>
        <li>Backpropagation</li>
        <li>Gradient Descent</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.code("""
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(X_train, y_train, epochs=50)
        """, language="python")

    # ---------------- STEP 9 ----------------
    st.markdown("""
        <div class="metal-card">
        <div class="metal-title">9. Loss Function & Optimization</div>
        <p class="metal-text"><b>Common Losses</b></p>
        <ul class="metal-list">
        <li>MSE → Regression</li>
        <li>Cross-entropy → Classification</li>
        </ul>
        <p class="metal-text"><b>Optimizers</b></p>
        <ul class="metal-list">
        <li>SGD</li>
        <li>Adam</li>
        <li>RMSProp</li>
        </ul>
        <div class="metal-strip">
        new_weight = old_weight - learning_rate × gradient
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- STEP 10 ----------------
    st.markdown("""
        <div class="metal-card">
        <div class="metal-title">10. Evaluation</div>
        <p class="metal-text"><b>ML Metrics</b></p>
        <ul class="metal-list">
        <li>Accuracy</li>
        <li>Precision</li>
        <li>Recall</li>
        <li>F1</li>
        <li>ROC-AUC</li>
        </ul>
        <p class="metal-text"><b>Regression Metrics</b></p>
        <ul class="metal-list">
        <li>MAE</li>
        <li>MSE</li>
        <li>RMSE</li>
        <li>R²</li>
        </ul>
        <p class="metal-text"><b>DL Metrics</b></p>
        <ul class="metal-list">
        <li>Same metrics monitored per epoch</li>
        </ul>
        <p class="metal-text"><b>Check:</b></p>
        <ul class="metal-list">
        <li>Overfitting</li>
        <li>Underfitting</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

#---------------Few Content on Deep Learning-----------------
def DeepLearning_Content():
    st.markdown("""
        <style>
        /* Background color as gree*/
        .stApp {
            background: transparent;
        }

        /* Soft Green Blurred Background */
        body::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;

            /* Soft gradient instead of hard green */
            background: linear-gradient(
                135deg,
                rgba(0, 90, 60, 0.7),
                rgba(0, 150, 90, 0.55),
                rgba(0, 70, 50, 0.7)
            );

            filter: blur(30px);   /* More blur = softer */
            transform: scale(1.1);
            z-index: -2;
        }

        /* Dark glass overlay */
        body::after {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.15);   /* lighter overlay */
            z-index: -1;
        }

        /* Bring Streamlit content forward */
        section.main, header, footer {
            position: relative;
            z-index: 1;
        }

        /* =============================
        INNER MINI CARD
        ============================= */
        .inner-card {
            background: linear-gradient(145deg,#062e1a,#064e2a);
            border-radius: 14px;
            padding: 15px 18px;
            margin-top: 15px;

            border: 1px solid rgba(74,222,128,0.35);

            box-shadow:
                0 0 12px rgba(74,222,128,0.25),
                inset 0 0 10px rgba(74,222,128,0.18);

            transition: all 0.3s ease;
        }
        
        /* =============================
        TEXT STYLE
        ============================= */
        .inner-card p {
            margin-bottom: 8px;
            font-size: 14px;
            line-height: 1.6;
        }


        /* =============================
        MODERN BULLET POINTS
        ============================= */
        .inner-card ul {
            list-style: none;
            padding-left: 0;
            margin-top: 10px;
        }

        .inner-card ul li {
            position: relative;
            padding-left: 28px;
            margin-bottom: 8px;
            font-size: 14px;
        }

        /* Neon bullet */
        .inner-card ul li::before {

            content: "";
            position: absolute;
            left: 0;
            top: 7px;

            width: 10px;
            height: 10px;

            border-radius: 50%;

            background: radial-gradient(circle,#4ade80,#22c55e);

            box-shadow:
                0 0 6px #22c55e,
                0 0 12px #22c55e;

        }

        /* Optional connecting line */
        .inner-card ul li::after {

            content: "";
            position: absolute;
            left: 4px;
            top: 18px;

            width: 2px;
            height: calc(100% + 4px);

            background: rgba(34,197,94,0.25);
        }

        .inner-card ul li:last-child::after {
            display: none;
        }

        </style>
        """, unsafe_allow_html=True)

    st.markdown("""
        <style>

        /* ===== MAIN BACKGROUND ===== */
        .stApp {
            background: transparent !important;
            color: #e5e7eb;
        }

        /* ===== CENTER TITLE ===== */
        .main-title {
            text-align: center;
            font-size: 50px;
            font-weight: 900;
            margin-bottom: 30px;
            background: linear-gradient(90deg,#22c55e,#4ade80,#86efac);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* ===== CARD GRID ===== */
        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 30px;
            padding: 20px;
        }

        /* ===== CARD STYLE ===== */
        .card {
            background: linear-gradient(145deg,#02140a,#041b10);
            border-radius: 18px;
            padding: 25px;
            min-height: 250px;

            border: 1px solid rgba(34,197,94,0.3);

            box-shadow:
                0 0 20px rgba(34,197,94,0.25),
                inset 0 0 15px rgba(34,197,94,0.15);

            transition: 0.4s ease;
        }

        /* ===== HOVER POP EFFECT ===== */
        .card:hover {
            transform: translateY(-12px) scale(1.03);
            box-shadow:
                0 0 40px rgba(34,197,94,0.6),
                0 0 80px rgba(34,197,94,0.4),
                inset 0 0 25px rgba(34,197,94,0.25);
        }

        /* ===== CARD TITLES ===== */
        .card h2 {
            text-align: center;
            margin-bottom: 15px;
            color: #4ade80;
        }

        /* ===== TEXT ===== */
        .card p, .card li {
            font-size: 15px;
            line-height: 1.6;
        }

        /* ===== LIST STYLE ===== */
        .card ul {
            padding-left: 20px;
        }

        </style>
        """, unsafe_allow_html=True)

        # =============================
        # TITLE
        # =============================
    st.markdown('<div class="main-title">Deep Learning Complete Guide</div>', unsafe_allow_html=True)

        # =============================
        # CONTENT CARDS
        # =============================
    st.markdown("""
        <div class="card-grid">

        <div class="card">
        <h2>What is Deep Learning?</h2>
        <div class="inner-card">
        <p>
        Deep Learning is a subset of Machine Learning that uses Artificial Neural Networks
        with multiple layers to learn patterns from data automatically.
        </p>

        <ul>
        <li>Inspired by human brain neurons</li>
        <li>Automatically extracts features</li>
        <li>Works well with large datasets</li>
        <li>Used in Computer Vision, NLP, Speech</li>
        </ul>
        </div>
        </div>
        
        <div class="card">
        <h2>How Deep Learning Works</h2>
        <div class="inner-card">
        <p>
        Deep learning models learn through iterative optimization.
        </p>

        <ul>
        <li>Input data enters the network</li>
        <li>Forward propagation computes outputs</li>
        <li>Loss function measures error</li>
        <li>Backpropagation computes gradients</li>
        <li>Optimizer updates weights</li>
        <li>Process repeats for many epochs</li>
        </ul>
        </div>
        </div>
        
        <div class="card">
        <h2>Model Creation</h2>
        <div class="inner-card">
        <p>
        A neural network is composed of layers:
        </p>

        <ul>
        <li>Input Layer</li>
        <li>Hidden Layers</li>
        <li>Output Layer</li>
        <li>Dense / Convolution / Recurrent layers</li>
        </ul>
        </div>
        </div>
        
        <div class="card">
        <h2>Activation Functions</h2>
        <div class="inner-card">
        <ul>
        <li>ReLU — most popular</li>
        <li>Sigmoid — probability output</li>
        <li>Tanh — centered output</li>
        <li>Softmax — multi-class classification</li>
        </ul>
        </div>
        </div>
        
        <div class="card">
        <h2>Loss Functions</h2>
        <div class="inner-card">
        <ul>
        <li>MSE — regression</li>
        <li>MAE — regression</li>
        <li>Binary Cross Entropy</li>
        <li>Categorical Cross Entropy</li>
        </ul>
        </div>
        </div>
        
        <div class="card">
        <h2>Optimizers</h2>
        <div class="inner-card">
        <ul>
        <li>SGD — basic gradient descent</li>
        <li>Adam — most widely used</li>
        <li>RMSprop</li>
        <li>Adagrad</li>
        </ul>
        </div>
        </div>
        
        <div class="card">
        <h2>Evaluation Metrics</h2>
        <div class="inner-card">
        <ul>
        <li>Accuracy</li>
        <li>Precision</li>
        <li>Recall</li>
        <li>F1 Score</li>
        <li>Confusion Matrix</li>
        </ul>
        </div>
        </div>
        
        <div class="card">
        <h2>Applications</h2>
        <div class="inner-card">
        <ul>
        <li>Image Recognition</li>
        <li>Self Driving Cars</li>
        <li>Chatbots</li>
        <li>Medical Diagnosis</li>
        <li>Recommendation Systems</li>
        </ul>
        </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

#---------------Few Content on Machine Learning--------------
def MachineLearning_Content():
    st.markdown("""
        <style>

        /* ===== APP BACKGROUND ===== */
        .stApp {
            background: linear-gradient(135deg, #020024, #090979, #000428);
            color: #e2e8f0;
        }

        /* ===== MAIN TITLE ===== */
        .ml-title {
            text-align: center;
            font-size: 48px;
            font-weight: 800;
            margin-bottom: 25px;
            background: linear-gradient(90deg,#a78bfa,#60a5fa,#22d3ee);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* ===== GRID ===== */
        .ml-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
            gap: 28px;
            padding: 20px;
        }

        /* ===== CARD ===== */
        .ml-card {
            backdrop-filter: blur(14px);
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            padding: 25px;
            min-height: 260px;

            border: 1px solid rgba(167,139,250,0.3);

            box-shadow:
                0 0 25px rgba(99,102,241,0.25),
                inset 0 0 15px rgba(255,255,255,0.05);

            transition: 0.35s ease;
        }

        /* Hover animation */
        .ml-card:hover {
            transform: translateY(-10px) scale(1.02);
            box-shadow:
                0 0 40px rgba(139,92,246,0.6),
                0 0 80px rgba(59,130,246,0.4);
        }

        /* ===== HEADINGS ===== */
        .ml-card h2 {
            text-align: center;
            margin-bottom: 14px;
            color: #c4b5fd;
        }
        .ml-card h3 {
            text-align: center;
            margin-bottom: 14px;
            color: #c4b5fd;
        }

        /* ===== TEXT ===== */
        .ml-card p {
            font-size: 15px;
            line-height: 1.6;
        }

        /* ===== LIST ===== */
        .ml-card ul {
            list-style: none;
            padding-left: 0;
            margin-top: 12px;
        }

        .ml-card ul li {
            position: relative;
            padding-left: 26px;
            margin-bottom: 8px;
            font-size: 14px;
        }

        /* Neon square bullet */
        .ml-card ul li::before {
            content: "";
            position: absolute;
            left: 0;
            top: 6px;

            width: 10px;
            height: 10px;

            border-radius: 3px;

            background: linear-gradient(135deg,#a78bfa,#60a5fa);

            box-shadow:
                0 0 6px #818cf8,
                0 0 12px #60a5fa;
        }

        </style>
        """, unsafe_allow_html=True)


        # =============================
        # TITLE
        # =============================
    st.markdown('<div class="ml-title">Machine Learning Complete Guide</div>', unsafe_allow_html=True)


        # =============================
        # CONTENT
        # =============================
    st.markdown("""

        <div class="ml-grid">

        <div class="ml-card">
        <h2>Problem Definition</h2>
        <p>
        This is the most critical step where the objective of the project is clearly defined.
        Understanding the business goal ensures correct model selection and evaluation.
        </p>
        <ul>
        <li>Define problem statement</li>
        <li>Identify inputs and outputs</li>
        <li>Determine success metrics</li>
        <li>Understand constraints</li>
        <li>Translate business → ML problem</li>
        </ul>
        </div>

        <div class="ml-card">
        <h2>Data Collection</h2>
        <p>
        Machine learning depends heavily on data. The quality and quantity of data directly
        impact model performance.
        </p>
        <ul>
        <li>Databases (SQL, NoSQL)</li>
        <li>APIs and web scraping</li>
        <li>Sensors / IoT devices</li>
        <li>Public datasets</li>
        <li>Data labeling and annotation</li>
        </ul>
        </div>

        <div class="ml-card">
        <h2>Data Preprocessing</h2>
        <p>
        Raw data is messy. Preprocessing converts data into a clean and usable format.
        </p>
        <ul>
        <li>Handling missing values</li>
        <li>Removing duplicates</li>
        <li>Encoding categorical variables</li>
        <li>Feature scaling (Normalization, Standardization)</li>
        <li>Outlier detection</li>
        </ul>
        </div>

        <div class="ml-card">
        <h3>Exploratory Data Analysis</h3>
        <p>
        EDA helps understand patterns, relationships, and distributions in data before modeling.
        </p>
        <ul>
        <li>Statistical summaries</li>
        <li>Correlation analysis</li>
        <li>Data visualization</li>
        <li>Feature relationships</li>
        <li>Detecting anomalies</li>
        </ul>
        </div>

        <div class="ml-card">
        <h2>Feature Engineering</h2>
        <p>
        Feature engineering improves model performance by transforming raw variables
        into meaningful features.
        </p>
        <ul>
        <li>Feature creation</li>
        <li>Feature selection</li>
        <li>Dimensionality reduction (PCA)</li>
        <li>Handling multicollinearity</li>
        <li>Domain knowledge features</li>
        </ul>
        </div>

        <div class="ml-card">
        <h2>Model Selection</h2>
        <p>
        Choosing the right algorithm depends on data size, complexity, and problem type.
        </p>
        <ul>
        <li>Regression models</li>
        <li>Classification models</li>
        <li>Clustering algorithms</li>
        <li>Tree-based methods</li>
        <li>Neural networks</li>
        </ul>
        </div>

        <div class="ml-card">
        <h2>Model Training</h2>
        <p>
        Training is the process where the model learns patterns from data by optimizing parameters.
        </p>
        <ul>
        <li>Train/Test split</li>
        <li>Cross validation</li>
        <li>Hyperparameter tuning</li>
        <li>Loss minimization</li>
        <li>Optimization algorithms</li>
        </ul>
        </div>

        <div class="ml-card">
        <h2>Model Evaluation</h2>
        <p>
        Evaluation measures how well the model performs on unseen data.
        </p>
        <ul>
        <li>Accuracy, Precision, Recall</li>
        <li>F1 Score</li>
        <li>ROC-AUC</li>
        <li>Confusion Matrix</li>
        <li>Regression metrics (MSE, RMSE, R²)</li>
        </ul>
        </div>

        <div class="ml-card">
        <h2>Model Deployment</h2>
        <p>
        Deployment makes the model available for real-world usage.
        </p>
        <ul>
        <li>REST APIs (FastAPI, Flask)</li>
        <li>Cloud deployment</li>
        <li>Batch predictions</li>
        <li>Edge deployment</li>
        <li>Integration with applications</li>
        </ul>
        </div>

        <div class="ml-card">
        <h3>Monitoring & Maintenance</h3>
        <p>
        After deployment, models must be monitored for performance degradation.
        </p>
        <ul>
        <li>Model drift detection</li>
        <li>Data drift monitoring</li>
        <li>Retraining pipelines</li>
        <li>Performance tracking</li>
        <li>Version control</li>
        </ul>
        </div>

        <div class="ml-card">
        <h3>Types of Machine Learning</h3>
        <ul>
        <li>Supervised Learning</li>
        <li>Unsupervised Learning</li>
        <li>Semi-Supervised Learning</li>
        <li>Reinforcement Learning</li>
        </ul>
        </div>

        <div class="ml-card">
        <h2>Common Algorithms</h2>
        <ul>
        <li>Linear Regression</li>
        <li>Logistic Regression</li>
        <li>Decision Trees</li>
        <li>Random Forest</li>
        <li>Support Vector Machines</li>
        <li>K-Means Clustering</li>
        <li>Gradient Boosting (XGBoost, LightGBM)</li>
        </ul>
        </div>

        </div>

        """, unsafe_allow_html=True)

with st.spinner('Training...'):
    #--------------Creating Data Set------------------ 
    if st.session_state.MODEL is None:
        home_content()       
    if st.session_state.DATA_OPT == "Create Data" and not st.session_state.DESIGN_MODEL:
        cd.Data_Creation()
    if st.session_state.DATA_OPT == 'Use Data' and not st.session_state.DESIGN_MODEL:
        cd.Use_Data()
    if st.session_state.DESIGN_MODEL and not st.session_state.START_TRAIN:
        NN.design_ANN()
    if st.session_state.START_TRAIN:
        NN.train_model()
    if st.session_state.MODEL == 'Deep Learning' and st.session_state.DATA_OPT is None:
        DeepLearning_Content()
    if st.session_state.MODEL == 'Machine Learning' and st.session_state.DATA_OPT is None:
        MachineLearning_Content()

    
    
        
    
    
    


