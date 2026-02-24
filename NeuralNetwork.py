import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

import torch as t
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as skPre
from Project.plot_decision_surface import plot_decision_boundary,plot_loss_curve,draw_network,plot_decision_surface_3D

SCALER_TECH = {
    'Standard Scaler':skPre.StandardScaler(),
    'Min-Max Scaler':skPre.MinMaxScaler(),
    'Robust Scaler':skPre.RobustScaler(),
    'Mean Absolute Scaler':skPre.MaxAbsScaler(),
    'Unit Vector Normalization':skPre.Normalizer(),
}
#---------------getting data,performing splitting,scaling------------
def data_splitter():
    global SCALER_TECH
    df = st.session_state.DATAFRAME
    # st.write(df)
    split = st.session_state.SPLIT_RATIO
    device = 'cuda' if st.session_state.TRAIN_DEVICE == 'GPU' and t.cuda.is_available() else 'cpu'
    scaling = st.session_state.SCALING
    #---------------DataFrame to Numpy---------------
    data = df.to_numpy()
    if data.shape[1] == 3:
        fv,cv = data[:,:2],data[:,2]
    else:
        fv,cv = data[:,:3],data[:,3]
    #---------------Splitting of data-----------------
    if st.session_state.DATA_TYPE == 'Classification':
        No_cls = df['Class Label'].nunique()
        x_train,x_test,y_train,y_test = train_test_split(fv,cv,train_size=split,shuffle=True,stratify=cv)
        x_trainf,x_cv,y_trainf,y_cv = train_test_split(x_train,y_train,train_size=split,shuffle=True,stratify=y_train)
    else:
        x_train,x_test,y_train,y_test = train_test_split(fv,cv,train_size=split,shuffle=True)
        x_trainf,x_cv,y_trainf,y_cv = train_test_split(x_train,y_train,train_size=split,shuffle=True)
    #-------------Performing scaling--------------------
    if scaling is not None:
        scaler = SCALER_TECH[scaling]
        x_trainf = scaler.fit_transform(x_trainf)
        x_cv = scaler.transform(x_cv)
        x_test = scaler.transform(x_test)
    #---------------changing to Tensors and selected device---------------
    dataType = t.long if st.session_state.DATA_TYPE == 'Classification' and No_cls>2 else t.float32
    x_trainf = t.from_numpy(x_trainf).to(dtype=t.float32,device=device)
    y_trainf = t.from_numpy(y_trainf).to(dtype=dataType,device=device)
    x_cv = t.from_numpy(x_cv).to(dtype=t.float32,device=device)
    y_cv = t.from_numpy(y_cv).to(dtype=dataType,device=device)
    x_test = t.from_numpy(x_test).to(dtype=t.float32,device=device)
    y_test = t.from_numpy(y_test).to(dtype=dataType,device=device)
        
    return (x_trainf,y_trainf,x_cv,y_cv,x_test,y_test)
        
#---------------Class for Neural Network-----------------
@st.cache_resource
class DesignModel(t.nn.Module):
    ACTIVATION_FUNC = {
        'Sigmoid':t.nn.Sigmoid(),
        'ReLU':t.nn.ReLU(),
        'LeakyReLU':t.nn.LeakyReLU(negative_slope=0.01),
        'RReLU':t.nn.RReLU(lower=0.125,upper=0.33),
        'GELU':t.nn.GELU(),
        'SiLU':t.nn.SiLU(),
        'Softplus':t.nn.Softplus(),
        'Tanh':t.nn.Tanh(),
        'Linear':t.nn.Identity(),
    }
    #-----------------designing the architecture of model--------------
    def __init__(self,In_Neuron,hidden_layer,No_Neuron_per_layer,Out_Neuron,active_func,Loss_func):
        super().__init__()
        Layers = []
        prev_neurons = In_Neuron
        for i in range(hidden_layer):
            Layers.extend([t.nn.Linear(in_features=prev_neurons,out_features=No_Neuron_per_layer[i],bias=True),
                            DesignModel.ACTIVATION_FUNC[active_func[i]]])
            prev_neurons = No_Neuron_per_layer[i]
        Layers.append(t.nn.Linear(in_features=prev_neurons,out_features=Out_Neuron,bias=True))
        if Loss_func in ['MSELoss','L1loss']:
            Layers.append(DesignModel.ACTIVATION_FUNC['Linear'])
        elif Loss_func == 'BCE':
            Layers.append(DesignModel.ACTIVATION_FUNC['Sigmoid'])
        self.Layers = t.nn.Sequential(*Layers)
    #--------------Performing weight initialization---------------------
        self.apply(self.init_weights)
    def init_weights(self,m):
        if isinstance(m,t.nn.Linear):
            t.nn.init.xavier_normal_(m.weight)
            t.nn.init.zeros_(m.bias)
    #-----------------forward propogation-------------------------------- 
    def forward(self,data):
        return self.Layers(data)

#-------------------performing epochs------------------------------------- 
LOSS_FUNC = {
    'MSELoss':t.nn.MSELoss(),
    'L1Loss':t.nn.L1Loss(),
    'BCE':t.nn.BCELoss(),
    'BCEwithlogitsloss':t.nn.BCEWithLogitsLoss(),
    'CrossEntropyLoss':t.nn.CrossEntropyLoss()
    } 
def train_model():
    global LOSS_FUNC
    st.title(":rainbow[Training Started]")
    device = 'cuda' if st.session_state.TRAIN_DEVICE == 'GPU' and t.cuda.is_available() else 'cpu'
    #-------------The Button logic-----------------
    def toggle_train():
        st.session_state.TRAIN_SWITH = not st.session_state.TRAIN_SWITH
    cols = st.columns([0.011,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],gap='xxsmall')
    with cols[0]:
        if st.button("🔄 Restart"):
            st.session_state.EPOCH = 0
            st.session_state.TRAIN_LOSS = 0
            st.session_state.TEST_LOSS = 0
            st.session_state.CV_LOSS = 0
            st.session_state.TRAIN_LOSS_PLOT = []
            st.session_state.TEST_LOSS_PLOT = []
            st.session_state.CV_LOSS_PLOT = []
    with cols[1]:
        st.button("▶️ Start" if not st.session_state.TRAIN_SWITH else "⏹️ Stop",on_click=toggle_train)
        
    #------------Calling the model------------  
    inputs = st.session_state.CREATE_MODEL
    myModel = DesignModel(inputs['Input_Neurons'],inputs['Layers'],inputs['Neurons'],inputs['Output_Neurons'],inputs['Activation'],inputs['Loss_Function']).to(device=device)
    Loss_Func = LOSS_FUNC[inputs['Loss_Function']]
    Optimizer = t.optim.SGD(params=myModel.parameters(),lr=inputs['learning_rate'])
    x_trainf,y_trainf,x_cv,y_cv,x_test,y_test = data_splitter()
    col1,col2,col3 = st.columns([0.01,0.01,0.01],gap='xxsmall')
    with col1:
        st.markdown(f"""<span style='font-size:20px;color:white;'>Model device:</span>
                    <span style="color:green;">{str(next(myModel.parameters()).device)[:-2]}</span>""",unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<span style='font-size:20px;color:white;'>Train Data device:</span>
                    <span style="color:green;">{str(x_trainf.device)[:-2]}</span>""",unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<span style='font-size:20px;color:white;'>CV Data device:</span>
                    <span style="color:green;">{str(x_cv.device)[:-2]}</span>""",unsafe_allow_html=True)
    col1,col2,col3 = st.columns([0.01,0.01,0.01],gap='xxsmall')
    with col1:
        space0 = st.empty() #The Space which will be used again and again
    with col2:
        space1 = st.empty()
    with col3:
        space2 = st.empty()
    st.divider()
    #--------------Plotting the decision surface and train loss line graph-------
    col1,col2= st.columns([0.5,0.5])
    with col1:
        Plot_space = st.empty()
        st.markdown("""
            <div style="margin-left:140px;
                        margin-right:180px;
                        ipadding-left:10px;
                        font-size:20px;">
                Decision Surface
            </div>
                    """,unsafe_allow_html=True)
    with col2:
        Plot_loss = st.empty()
        st.markdown("""
            <div style="margin-left:140px;
                        margin-right:180px;
                        ipadding-left:10px;
                        font-size:20px;">
                Train/Test Loss
            </div>
                    """,unsafe_allow_html=True)
    while st.session_state.TRAIN_SWITH:
        #-------------model training---------------
        train_pre_y = myModel(x_trainf)
        train_loss = Loss_Func(train_pre_y.squeeze_(),y_trainf)
        Optimizer.zero_grad()
        train_loss.backward()
        Optimizer.step()
        #---------------cv testing-----------------
        myModel.eval()
        with t.no_grad():
            cv_pre_y = myModel(x_cv)
            cv_loss = Loss_Func(cv_pre_y.squeeze_(),y_cv)
        #-------------Epoch Report------------------
        space0.markdown("""
            <div style='font-size:25px;color:orange;'>
                Epoch: {:,}
            </div>
            """.format(st.session_state.EPOCH+1),unsafe_allow_html=True)
        #-------------Train Test Loss Report----------
        space1.markdown("""
            <div style='font-size:25px;color:orange;'>
                Train Loss: {:.6f}
            </div>
            """.format(train_loss),unsafe_allow_html=True)
        space2.markdown("""
            <div style='font-size:25px;color:orange;'>
                CV Loss: {:.6f}
            </div>
            """.format(cv_loss),unsafe_allow_html=True)
        if device == 'cuda':
            st.session_state.CV_LOSS_PLOT.append(cv_loss.cpu().numpy().item())
            st.session_state.TRAIN_LOSS_PLOT.append(train_loss.detach().cpu().numpy().item())
        else:
            st.session_state.CV_LOSS_PLOT.append(cv_loss.numpy().item())
            st.session_state.TRAIN_LOSS_PLOT.append(train_loss.detach().numpy().item())
            
        st.session_state.CV_LOSS = cv_loss.item()
        st.session_state.TRAIN_LOSS = train_loss.item()
        st.session_state.EPOCH += 1
        
    #--------------Plotting the decision surface and train loss line graph-------
        if st.session_state.EPOCH % 10 == 0:
            if x_trainf.shape[1] == 2:
                st.session_state.DPLOT = plot_decision_boundary(myModel,x_trainf,y_trainf)
                Plot_space.pyplot(st.session_state.DPLOT)
            elif x_trainf.shape[1] == 3:
                st.session_state.DPLOT = plot_decision_surface_3D(myModel,x_trainf,y_trainf)
                Plot_space.pyplot(st.session_state.DPLOT)
            #-------------LineGraph-------------------------------
            st.session_state.LPLOT = plot_loss_curve(st.session_state.TRAIN_LOSS_PLOT,st.session_state.CV_LOSS_PLOT)
            Plot_loss.pyplot(st.session_state.LPLOT)
                
        
    #-------------Epoch Report------------------
    space0.markdown("""
        <div style='font-size:25px;color:orange;'>
            Epoch: {:,}
        </div>
        """.format(st.session_state.EPOCH),unsafe_allow_html=True)
    space1.markdown("""
        <div style='font-size:25px;color:orange;'>
            Train Loss: {:.6f}
            </div>
        """.format(st.session_state.TRAIN_LOSS),unsafe_allow_html=True)
    space2.markdown("""
        <div style='font-size:25px;color:orange;'>
            CV Loss: {:.6f}
        </div>
        """.format(st.session_state.CV_LOSS),unsafe_allow_html=True)
    #---------------Decision surface-------------------
    if st.session_state.DPLOT is not None:
        Plot_space.pyplot(st.session_state.DPLOT)  
        #--------------Loss Line----------------------------
        Plot_loss.pyplot(st.session_state.LPLOT)
    # st.session_state.MY_MODEL = myModel

#-----------------Designing the ANN Architecture--------------
def design_ANN():
    with st.spinner('Training...'):
        st.subheader(":rainbow[Design Deep Learning Model]")
        col1, col2,col3 = st.columns([0.01,0.01,0.01],gap='xxsmall')
        with col1:
            hidden_layer = st.number_input("No of Hidden Layers",min_value=1,max_value=100,value=1,step=1) 
        with col2:
            Loss_func = st.selectbox("Loss Function",['MSELoss','L1Loss','BCE','BCEwithlogitsloss','CrossEntropyLoss'])
        with col3:
            learing_rate = st.number_input("Learning Rate",min_value=0.01,max_value=100.0,value=0.01,step=0.01)
        col1, col2 = st.columns([0.01,0.01],gap='xxsmall')
        with col1:
            input_neurons = st.number_input("Input Neurons",1,10,1)
        with col2:
            output_neurons = st.number_input("Output Neurons", 1, 10, 1)
        
        rows =  (hidden_layer - 1) // 3 + 1
        no_of_neurons = []
        active_fnc = []
        layer_index = 0
        for r in range(rows):
            cols = st.columns(int(np.clip(hidden_layer-layer_index,1,3)),gap='xxsmall')  
            for c in range(3):
                if layer_index < hidden_layer:
                    with cols[c]:   
                        no_of_neurons.append(st.number_input(f"Neurons in Hidden Layer {layer_index+1}", 1, 10, 4))
                        active_fnc.append(st.selectbox(f"Activation Function of Layer {layer_index+1}",
                                    ['Sigmoid','ReLU','LeakyReLU','RReLU','GELU','SiLU','Softplus','Tanh','Linear']))
                layer_index +=1
                
        architecture = [input_neurons] + no_of_neurons + [output_neurons]       
        col1,col2,col3 = st.columns([0.01,0.01,0.01])

        with col1:
            if st.button("Create AI Model"):
                st.session_state.CREATE_MODEL = {'learning_rate':learing_rate,'Activation':active_fnc,'Layers':hidden_layer,
                                                    'Neurons':no_of_neurons,'Input_Neurons':input_neurons,'Output_Neurons':output_neurons,
                                                    'Loss_Function':Loss_func}
                fig = draw_network(architecture)
                st.session_state.SIDE_MODEL_FIG = fig
                st.session_state.SIDE_MODEL_PLOT = True
        if st.session_state.SIDE_MODEL_PLOT:
            with col2:
                st.button('Preview')
        st.subheader(":rainbow[Neural Network Visualizer]")
        with st.spinner("Plotting..."):
            fig = draw_network(architecture)
            st.pyplot(fig,use_container_width=False)


    
