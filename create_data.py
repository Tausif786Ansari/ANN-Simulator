import streamlit as st
from sklearn.datasets import make_classification,make_regression
import random as rd
import torch as t
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import code_create_data as ccd
#--------------------Plotting the points----------------
def plot_data_nav(Nav_data):
    df = st.session_state.DATAFRAME
    if Nav_data is None: 
        design = st.session_state.DATA_DESIGN
        dtype =  st.session_state.DATA_TYPE
        no_cls = st.session_state.CREATE_DATA.get('No_Class',None)
        if no_cls is not None:
            No_class = 2 if design == 'Board' else no_cls
    else:
        design = Nav_data['DATA_DESIGN']
        dtype = Nav_data['DATA_TYPE']
        if Nav_data['No_Class'] is not None:
            No_class = 2 if design == 'Board' else Nav_data['No_Class']
        
    if design in ['3D Spiral','3D Vessel']:
        fig = plt.figure(figsize=(6,4), dpi=120)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs=df['Feature1'],ys=df['Feature2'],zs=df['Feature3'],
                   c=df['Class Label'],cmap='coolwarm',s=15)
        ax.set(xticklabels=[],yticklabels=[],zticklabels=[])
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=25, azim=45)
        fig.subplots_adjust(0,0,1,1)
    else:
        fig,ax = plt.subplots(figsize=(6,4))
        if dtype == 'Classification':
            color = {0:'red',1:'orange',2:'blue',3:'green',4:'pink',5:'violet',6:'black',7:'gold',8:'cyan',9:'olive',10:'mustard'}
            sns.scatterplot(x=df['Feature1'],y=df['Feature2'],hue=df['Class Label'],
                            s=18,ax=ax,palette=dict(list(color.items())[:No_class]),legend=False)
        else:
            sns.scatterplot(x=df['Feature1'],y=df['Target'],s=18,ax=ax,legend=False)
        ax.tick_params(labelsize=6.0,color='white',labelcolor='white')
    ax.set(xlabel=None,ylabel=None)
    ax.set_facecolor((0.0, 0.8, 0.6, 1))
    fig.set_facecolor("black")
    return fig

def Data_Creation():
    st.title(":rainbow[Create Your Own DataSet]")
    data_type = st.segmented_control("Select Type of Data",['Classification','Regression'],selection_mode='single',default=None)
    #--------------Reset Values-----------------------
    if data_type != st.session_state.DATA_TYPE:
        st.session_state.DATA_TYPE = data_type
        st.session_state.CREATE_DATA = {}
        st.session_state.DATA_CREATED = False
        st.session_state.PREVIEW_DATA = False
        st.session_state.DATA_DESIGN = None
    #------------Create Classification Data---------------
    if st.session_state.DATA_TYPE == 'Classification':
        st.session_state.DATA_DESIGN = st.segmented_control("Select Design of Data",['Wave','Star','3D Spiral','3D Vessel','Mobius','Hill','Ring','Board'],selection_mode='single',default=None)
        n_row=st.number_input("Enter Number of Rows",min_value=100,max_value=1000000,step=100,placeholder=100)
        noise=st.slider("Noise", min_value=0.0, max_value=1.0, step=0.01)
        if st.session_state.DATA_DESIGN == 'Board':
            n_grid=st.number_input("No of Grids",min_value=0.1,max_value=5.0,step=0.01,placeholder=0.01)
        else:
            n_class=st.number_input("No of Classes",min_value=2,max_value=10,step=1,placeholder=2)
        if st.session_state.DATA_DESIGN is not None:
            cls_sep = st.slider("Class Separater", min_value=0.1, max_value=0.5, step=0.01)
        else:
            cls_sep = st.slider("Class Separater", min_value=0.1, max_value=5.0, step=0.1)
        if st.session_state.DATA_DESIGN is None:
            ncps=st.number_input("No of Cluster Per Class",min_value=1,max_value=10,step=1,placeholder=1)
            col1,col2 = st.columns([0.01,0.01],gap='xxsmall')
            weights=[0.5,0.5]
            with col1:
                weights[0] = st.number_input("Weight Given to First Class",min_value=0.0,max_value=1.0,step=0.01,placeholder=0.5)
            with col2:
                weights[1]=st.number_input("Weight Given to Second Class",min_value=0.0,max_value=1.0,step=0.01,placeholder=0.5)
        random_state=st.number_input("Controlling The Randomness any integer value",min_value=0,max_value=1000000,step=1,placeholder=5)
        #--------------Updateing the values---------------------
        if st.session_state.DATA_DESIGN is None:
            st.session_state.CREATE_DATA = {'No_Rows':n_row,'Noise':noise,'No_Class':n_class,'Class_Sep':cls_sep,'ClusterPerCls':ncps,'Weight':weights,'random_state':random_state}
        elif st.session_state.DATA_DESIGN == 'Board':
            st.session_state.CREATE_DATA = {'No_Rows':n_row,'Noise':noise,'No_Class':n_grid,'Class_Sep':cls_sep,'random_state':random_state}
        else:
            st.session_state.CREATE_DATA = {'No_Rows':n_row,'Noise':noise,'No_Class':n_class,'Class_Sep':cls_sep,'random_state':random_state}
    #--------------Create Regression Data------------------
    if st.session_state.DATA_TYPE == 'Regression':
        n_row=st.number_input("Enter Number of Rows",min_value=100,max_value=1000000,step=100,placeholder=100,value=100)
        noise = st.slider("Noise", 0.0, 10.0, 1.0)
        st.session_state.CREATE_DATA = {'No_Rows':n_row,"Noise": noise}
    
    if st.session_state.DATA_TYPE:
        col1,col2,col3 = st.columns(3)
        with col1:
            #---------------Button to create the data_df-----------------
            if st.button("Create Dataset"):
                with st.spinner("Generating Dataset...."):
                    if st.session_state.DATA_TYPE == 'Classification':
                        data_design = st.session_state.DATA_DESIGN
                        match data_design:
                            case None:
                                #-----------------Classification Data----------------------
                                fv,cv = make_classification(n_samples=st.session_state.CREATE_DATA['No_Rows'],
                                            n_features=2,n_informative=2,n_redundant=0,n_repeated=0,
                                            n_classes=st.session_state.CREATE_DATA['No_Class'],
                                            n_clusters_per_class=st.session_state.CREATE_DATA['ClusterPerCls'],
                                            weights=st.session_state.CREATE_DATA['Weight'],
                                            random_state=st.session_state.CREATE_DATA['random_state'],
                                            class_sep=st.session_state.CREATE_DATA['Class_Sep'])
                                df_data = pd.DataFrame(fv,columns=['Feature1','Feature2'])
                                df_data['Class Label'] = cv
                            case 'Wave':
                                #----------------Sin Data----------------------------------
                                fv,cv = ccd.multi_sine(n_samples=st.session_state.CREATE_DATA['No_Rows'],
                                            noise=st.session_state.CREATE_DATA['Noise'],
                                            n_classes=st.session_state.CREATE_DATA['No_Class'],
                                            gap=st.session_state.CREATE_DATA['Class_Sep'],
                                            random_state=st.session_state.CREATE_DATA['random_state'])
                                df_data = pd.DataFrame(fv,columns=['Feature1','Feature2'])
                                df_data['Class Label'] = cv 
                            case 'Star':
                                #-----------------Star Design-------------------------------
                                fv,cv = ccd.radial_wedges(n_samples=st.session_state.CREATE_DATA['No_Rows'],
                                            noise=st.session_state.CREATE_DATA['Noise'],
                                            n_classes=st.session_state.CREATE_DATA['No_Class'],
                                            gap=st.session_state.CREATE_DATA['Class_Sep'],
                                            random_state=st.session_state.CREATE_DATA['random_state'])
                                df_data = pd.DataFrame(fv,columns=['Feature1','Feature2'])
                                df_data['Class Label'] = cv    
                            case '3D Spiral':
                                #------------------3D Dataset-------------------------------
                                fv,cv = ccd.helix_3d(n_samples=st.session_state.CREATE_DATA['No_Rows'],
                                            noise=st.session_state.CREATE_DATA['Noise'],
                                            random_state=st.session_state.CREATE_DATA['random_state'])
                                df_data = pd.DataFrame(fv,columns=['Feature1','Feature2','Feature3'])
                                df_data['Class Label'] = cv  
                            case '3D Vessel':
                                #-------------------3D Vessel-----------------------------
                                fv,cv = ccd.torus_3d(n_samples=st.session_state.CREATE_DATA['No_Rows'],
                                            noise=st.session_state.CREATE_DATA['Noise'],
                                            random_state=st.session_state.CREATE_DATA['random_state'])
                                df_data = pd.DataFrame(fv,columns=['Feature1','Feature2','Feature3'])
                                df_data['Class Label'] = cv  
                            case 'Mobius':
                                #------------------Donut shape----------------------------
                                fv,cv = ccd.mobius(n_samples=st.session_state.CREATE_DATA['No_Rows'],
                                            noise=st.session_state.CREATE_DATA['Noise'],
                                            gap=st.session_state.CREATE_DATA['Class_Sep'],
                                            random_state=st.session_state.CREATE_DATA['random_state'])
                                df_data = pd.DataFrame(fv,columns=['Feature1','Feature2'])
                                df_data['Class Label'] = cv  
                            case 'Hill':
                                #----------------------Sea Wave---------------------------
                                fv,cv = ccd.piecewise_linear(n_samples=st.session_state.CREATE_DATA['No_Rows'],
                                            noise=st.session_state.CREATE_DATA['Noise'],
                                            gap=st.session_state.CREATE_DATA['Class_Sep'],
                                            random_state=st.session_state.CREATE_DATA['random_state'])
                                df_data = pd.DataFrame(fv,columns=['Feature1','Feature2'])
                                df_data['Class Label'] = cv
                            case 'Ring':
                                #---------------------RING Shape--------------------------
                                fv,cv = ccd.parity_rings(n_samples=st.session_state.CREATE_DATA['No_Rows'],
                                            noise=st.session_state.CREATE_DATA['Noise'],
                                            gap=st.session_state.CREATE_DATA['Class_Sep'],
                                            random_state=st.session_state.CREATE_DATA['random_state'])
                                df_data = pd.DataFrame(fv,columns=['Feature1','Feature2'])
                                df_data['Class Label'] = cv
                            case 'Board':
                                #------------------Check Board------------------------
                                fv,cv = ccd.checkerboard(n_samples=st.session_state.CREATE_DATA['No_Rows'],
                                            noise=st.session_state.CREATE_DATA['Noise'],
                                            grid_size=st.session_state.CREATE_DATA['No_Class'],
                                            gap=st.session_state.CREATE_DATA['Class_Sep'],
                                            random_state=st.session_state.CREATE_DATA['random_state'])
                                df_data = pd.DataFrame(fv,columns=['Feature1','Feature2'])
                                df_data['Class Label'] = cv
                    elif st.session_state.DATA_TYPE == 'Regression':
                        #---------------------Making Regression Data-------------------
                        fv,cv = make_regression(n_samples=st.session_state.CREATE_DATA['No_Rows'],
                                    n_features=2,n_informative=2,n_targets=1,
                                    noise=st.session_state.CREATE_DATA['Noise'])
                        df_data = pd.DataFrame(fv,columns=['Feature1','Feature2'])
                        df_data['Target'] = cv
        
                    st.session_state.DATAFRAME = df_data
                    st.session_state.DATA_CREATED = True
                    st.session_state.PREVIEW_DATA = False
                    st.session_state.PLOT_DATA_NAV = {'DATA_TYPE':st.session_state.DATA_TYPE,
                            'No_Class':st.session_state.CREATE_DATA['No_Class'] if st.session_state.DATA_TYPE == 'Classification' else None,
                            'DATA_DESIGN':st.session_state.DATA_DESIGN}
                    fig = plot_data_nav(st.session_state.PLOT_DATA_NAV)
                    st.session_state.SIDEBAR_FIG = fig
                    st.session_state.SHOW_SIDEBAR_PLOT = True
        
        #---------------Successfully created data-------------------------
        if st.session_state.DATA_CREATED:
            if st.session_state.DATA_TYPE == 'Classification':
                st.success("Successfully created Classification Data")
            else:
                st.success("Successfully created Regression Data")
                    
        #-----------Preview Data------------------
        with col2:
            if st.session_state.DATA_CREATED:
                if st.button("Preview Data"):
                    st.session_state.PREVIEW_DATA = True
        #-------------Download button---------------
        with col3:
            if st.session_state.DATA_CREATED:
                csv = st.session_state.DATAFRAME.to_csv(index=False)
                st.download_button("Download Dataset",data=csv,file_name="Made_Data.csv",mime="text/csv")
        
        
        #------------Preview Data-------------
        if st.session_state.PREVIEW_DATA:
            with st.spinner("Generating..."):
                st.dataframe(st.session_state.DATAFRAME,
                            width='stretch',height='stretch',
                            use_container_width=True,hide_index=True)
            with st.spinner("Plotting..."):
                fig = plot_data_nav(None)
                st.pyplot(fig, use_container_width=True)
            
def Use_Data():
    def plot_data(data_df):
        if data_df is not None:
            fv,cv = data_df[:,:2],data_df[:,2]
            df_data = pd.DataFrame(fv,columns=['Feature1','Feature2'])
            df_data['Class Label'] = cv
            st.session_state.DATAFRAME = df_data
            st.dataframe(st.session_state.DATAFRAME,
                    width='stretch',height='stretch',
                    use_container_width=True,hide_index=True)
            with st.spinner("Plotting..."):
                fig,ax = plt.subplots(figsize=(6,4))
                sns.scatterplot(x=st.session_state.DATAFRAME['Feature1'],y=st.session_state.DATAFRAME['Feature2'],hue=st.session_state.DATAFRAME['Class Label'],s=18,ax=ax,
                                palette={0:'blue',1:'orange',2:'red'} if st.session_state.DATAFRAME['Class Label'].nunique() > 2 else None,legend=False)
                ax.set(xlabel=None,ylabel=None)
                ax.tick_params(labelsize=6.0,color='white',labelcolor='white')
                ax.set_facecolor((0.0, 0.8, 0.6, 1))
                fig.set_facecolor("black")
                st.pyplot(fig,width='content')
    st.title(":rainbow[Use The Predefined Data]")
    data_df = None
    match st.session_state.DATASET:
        case 'Circle':
            data_df = t.load(r"D:\language\ML-DL projects\FirstModel\Project\Dataset\circle_data.pt")
        case 'Swiss':
            data_df = t.load(r"D:\language\ML-DL projects\FirstModel\Project\Dataset\SwissRoll.pt")
        case 'XOR':
            data_df = t.load(r"D:\language\ML-DL projects\FirstModel\Project\Dataset\XOR_data.pt")
        case 'Moon':
            data_df = t.load(r"D:\language\ML-DL projects\FirstModel\Project\Dataset\moon_data.pt")
        case 'Disc':
            data_df = t.load(r"D:\language\ML-DL projects\FirstModel\Project\Dataset\Disc.pt")
        case 'Binary':
            data_df = t.load(r"D:\language\ML-DL projects\FirstModel\Project\Dataset\Binary.pt")
        case 'Spiral':
            data_df = t.load(r"D:\language\ML-DL projects\FirstModel\Project\Dataset\Spiral_data.pt")
        case 'Square':
            data_df = t.load(r"D:\language\ML-DL projects\FirstModel\Project\Dataset\Square_data.pt")
        case 'Board':
            data_df = t.load(r"D:\language\ML-DL projects\FirstModel\Project\Dataset\Board_data.pt")
        case 'Sine':
            data_df = t.load(r"D:\language\ML-DL projects\FirstModel\Project\Dataset\Sine_data.pt")
        case 'Multi_Swiss':
            data_df = t.load(r"D:\language\ML-DL projects\FirstModel\Project\Dataset\Multi_Swiss.pt")
        case 'Ring':
            data_df = t.load(r"D:\language\ML-DL projects\FirstModel\Project\Dataset\Ring.pt")
    with st.spinner("Plotting..."):
        plot_data(data_df)
        
        


                