import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
import random as rd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
COLOR_SURFACE = ['viridis','plasma','turbo','Purples','cividis']
def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):

    device = next(model.parameters()).device   # original device

    # Create CPU copy ONLY for plotting
    model_cpu = model.to("cpu")

    X_cpu, y_cpu = X.to("cpu"), y.to("cpu")

    # mesh grid
    x_min, x_max = X_cpu[:, 0].min() - 0.1, X_cpu[:, 0].max() + 0.1
    y_min, y_max = X_cpu[:, 1].min() - 0.1, X_cpu[:, 1].max() + 0.1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 300)
    )

    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    model_cpu.eval()
    with torch.inference_mode():
        y_logits = model_cpu(X_to_pred_on)

    if len(torch.unique(y_cpu)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))

    y_pred = y_pred.reshape(xx.shape).detach().numpy()


    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=200)
    # Small padding between canvas and plot
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
    
    n_classes = len(np.unique(y_pred))
    colors = plt.cm.turbo(np.linspace(0.3, 0.9, n_classes))
    green_cmap = ListedColormap(colors)

    ax.contourf(xx, yy, y_pred, cmap=green_cmap, alpha=0.7, antialiased=True)

    color = {
        0:'red',1:'orange',2:'blue',3:'green',4:'pink',
        5:'violet',6:'black',7:'gold',8:'cyan',9:'olive',10:'#FFDB58'
    }

    y_np = y_cpu.numpy().astype(int)
    point_colors = [color[i] for i in y_np]

    ax.scatter(X_cpu[:, 0],X_cpu[:, 1],c=point_colors,s=18,edgecolors="white",linewidth=0.2)

    # Thin border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_color("black")

    ax.margins(0)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0)
    # MOVE MODEL BACK TO ORIGINAL DEVICE
    model.to(device)
    return fig


def plot_decision_surface_3D(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):

    device = next(model.parameters()).device

    model_cpu = model.to("cpu")
    X_cpu, y_cpu = X.to("cpu"), y.to("cpu")

    # ---------- Mesh grid (feature 1 & 2) ----------
    x_min, x_max = X_cpu[:, 0].min() - 0.1, X_cpu[:, 0].max() + 0.1
    y_min, y_max = X_cpu[:, 1].min() - 0.1, X_cpu[:, 1].max() + 0.1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 120),
        np.linspace(y_min, y_max, 120)
    )

    # ---------- Fix third feature ----------
    z_fixed = X_cpu[:, 2].mean().item()

    grid = np.column_stack((xx.ravel(),yy.ravel(),np.full(xx.ravel().shape, z_fixed)))

    X_to_pred_on = torch.from_numpy(grid).float()

    # ---------- Prediction ----------
    model_cpu.eval()
    with torch.inference_mode():
        y_logits = model_cpu(X_to_pred_on)

    if len(torch.unique(y_cpu)) > 2:
        probs = torch.softmax(y_logits, dim=1).max(dim=1).values
    else:
        probs = torch.sigmoid(y_logits).squeeze()

    zz = probs.reshape(xx.shape).detach().numpy()

    # ---------- Plot ----------
    fig = plt.figure(figsize=(4, 3.5), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        xx, yy, zz,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.85
    )

    # Scatter points
    color_dict = {
        0:'red',1:'orange',2:'blue',3:'green',4:'pink',
        5:'violet',6:'black',7:'gold',8:'cyan',9:'olive',10:'#FFDB58'
    }

    y_np = y_cpu.numpy().astype(int)
    point_colors = [color_dict[i] for i in y_np]

    ax.scatter(
        X_cpu[:, 0],
        X_cpu[:, 1],
        X_cpu[:, 2],   # real Z values
        c=point_colors,
        s=18,
        edgecolors="white",
        linewidth=0.2
    )

    # Clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.view_init(elev=30, azim=135)

    plt.tight_layout(pad=0.5)

    model.to(device)

    return fig

def plot_loss_curve(train_loss, cv_loss):

    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)

    # Small padding between canvas and subplot (same as decision plot)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.12)

    # Plot lines
    sns.lineplot(x=epochs,y=train_loss,ax=ax,label='Train Loss') 
    sns.lineplot(x=epochs,y=cv_loss,ax=ax,label='CV Loss')

    # Labels
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Train/Test Loss", fontsize=8)
    #Ticks
    ax.tick_params(labelsize=5.0,labelrotation=60.0)

    # Thin border (same style)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_color("black")

    # Light grid
    ax.grid(alpha=0.15, linewidth=0.5)

    # Legend
    ax.legend(fontsize=7, frameon=False)

    # Clean margins
    ax.margins(x=0.02, y=0.05)

    return fig

#--------------function to draw ANN Design----------------
def draw_network(layer_sizes):
    #----------------Node Color--------------------
    def get_color(layer_index, total_layers):
        
        if layer_index == 0:
            return "#45e34d"   # input (blue)

        elif layer_index == total_layers - 1:
            return "#f6e30e"   # output (yellow)

        else:
            return "#ee0b0b"   # hidden (green)
       
    fig, ax = plt.subplots(figsize=(10, 6),facecolor='black')

    n_layers = len(layer_sizes)
    v_spacing = 1
    h_spacing = 2

    node_positions = []

    #------------Unique colors---------------
    bright_colors = plt.cm.tab10(np.linspace(0, 1, 10))   # bold
    dark_colors   = plt.cm.Dark2(np.linspace(0, 1, 8))    # dark
    colors = np.vstack([bright_colors,dark_colors])
    color_index = 0
    node_colors = {}   # store color for each node
    
    # Draw nodes
    for i, layer_size in enumerate(layer_sizes):

        layer_nodes = []
        y_positions = np.linspace(-layer_size/2, layer_size/2, layer_size)
        c = rd.choice(colors)
        for j,y in enumerate(y_positions):
            x = i * h_spacing
            circle = plt.Circle((x, y), 0.25, color=get_color(i, n_layers))
            min_circle = plt.Circle((x, y), 0.10, color=c)
            ax.add_patch(circle)
            ax.add_patch(min_circle)
            layer_nodes.append((x, y))
            if i == 0:
                ax.text(x - 0.6,y,f"f{j+1}",fontsize=12,ha="right",va="center",fontweight="bold",color='white')
            if i == n_layers - 1:
                ax.text(x + 0.6,y,f"y{j+1}",fontsize=12,ha="left",va="center",fontweight="bold",color='white')
        x_layer = i * h_spacing
        y_bottom = -3.0
        if n_layers < 10:
            if i == 0:
                ax.text(x_layer,y_bottom,"In L",fontsize=12,ha="center",va="center",fontweight="bold",color='white')
            elif i == n_layers - 1:
                ax.text(x_layer,y_bottom,"Out L",fontsize=12,ha="center",va="center",fontweight="bold",color='white')      
            else:
                ax.text(x_layer,y_bottom,f"HL {i}",fontsize=12,ha="center",va="center",fontweight="bold",color='white')  
        
        node_positions.append(layer_nodes)
    
    # Assign colors to nodes
    for layer_nodes in node_positions:
        for node in layer_nodes:
            node_colors[tuple(node)] = colors[color_index % len(colors)]
            color_index += 1

    # Draw connections
    for i in range(n_layers - 1):
        for node1 in node_positions[i]:
            for node2 in node_positions[i + 1]:
                color = node_colors[tuple(node1)]  # color based on source node
                ax.plot(
                    [node1[0], node2[0]],
                    [node1[1], node2[1]],
                    color=color,
                    linewidth=1.5,
                    alpha=0.9
                )

    ax.set_aspect('equal')
    ax.axis('off')

    return fig
