# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:13:23 2024

@author: bmillet
"""

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


# Reference the region based on the index of the 'fs_3d' field
dic_ocim = {'AA': 0, 'HS': 1, 'NP': 2, 'NA': 3, 'LLH': 4}
dyes = ['DyeLL', 'DyeMS',  'DyeNP', 'DyeHS', 'DyeNA', 'DyeAA']
dyes_TMI = ['dyeLL', 'dyeMS',  'dyeNP', 'dyeHS', 'dyeNA', 'dyeAA']
panels_letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w']
panels_letter_parenthesis = ['(' + letter + ')' for letter in panels_letter]

# Function that creates a list of 45 depth levels
def create_l_depth():
    l_depth0 = []
    for i in range(6):
        l_depth0.append(i*10)
    for i in range(6):
        l_depth0.append(75+i*25)
    for i in range(3):
        l_depth0.append(250+i*50)
    for i in range(11):
        l_depth0.append(400+i*100)
    for i in range(19):
        l_depth0.append(1500+i*250)
    return (l_depth0)



def approx_depth(d, l_depth):
    ''' Return the closest value in the list to the value given and the index associated to this value. '''
    for i in range(0,len(l_depth)):
        diff = abs(d-l_depth[i])
        if i ==0:
            min = (diff,i)
        elif min[0] > diff:
            min = (diff,i)
    return((l_depth[min[1]],min[1]))




# Important information about the different datasets
coords_dataset = [(-40, -38), (-40, -55), (-60, -52), (-60, -78), (-40, -72), (0, -50), (20, -50), (35, -40), (50, -38), (50, -57), (80, -58), (-170, -50)]
l_neut_dens = [27.96, 28.01, 27.93, 28.18,
              28.14, 28.03, 28.07, 28.02,
              28.03, 28.05, 28.07, 27.99]
l_index = [0, 3, 4, 5, 7, 9, 10]
l_names_accepted = ['WOCE A11 (Meredith et al 1999)', 'GEOSECS Ostlund et al (1987)', "Drake's Passage (Meredith et al 1999)", 'Weppernig et al (1996)', 
                'Mackensen et al (1996)', 'Mackensen et al (1996)', 'ADOX1 (Frew et al 1995)', 'SWINDEX (K.J. Heywood pers. communication)', 
                'ADOX2 (Frew et al 1995)', 'ADOX1 (Frew et al 1995)', 'ADOX1 (Frew et al 1995)', 'GEOSECS Ostlund et al (1987)']
# l_lon_accepted = [[(10, 40), (50, 80), (80, 110)], [(50, 80)], [(-70, -10)], [(-70, 30), (160, 270)], [(-10, 30), (-60, -10)], [(10, 60)], [(-60, 20)], [(-60, -30)]]
l_lon_accepted = [(-60, 20), (-70, 30), (-70, -10), (-60, -30),
                 (-60, -10), (-10, 30), (10, 40), (10, 60),
                  (50, 80), (50, 80), (80, 110), (160, 270)]

def create_dfs(aux):
    '''Takes the d18o SO compilation and separates it into 12 different datasets separated from the laboratories where they were produced as well as their geographical location.'''
    
    
    dfs = []; l_names_dfs = []
    for i in range(len(l_names_accepted)):
        lon_min, lon_max, name = l_lon_accepted[i][0], l_lon_accepted[i][1], l_names_accepted[i]
        l_names_dfs.append(name)
        if (lon_min < 0) & (lon_max < 0):
            dfs.append(aux.where((aux['Reference'] == name) & (aux['Longitude'] >= 360 + lon_min) & (aux['Longitude'] <= 360 + lon_max)).dropna())
        elif (lon_min < 0):
            dfs.append(aux.where((aux['Reference'] == name) & ((aux['Longitude'] >= 360 + lon_min) | (aux['Longitude'] <= lon_max))).dropna())
        else:
            dfs.append(aux.where((aux['Reference'] == name) & (aux['Longitude'] >= lon_min) & (aux['Longitude'] <= lon_max)).dropna())
            
    return dfs
    

    
def get_BoundNorm(vmin, vmax, cmapname = 'coolwarm', nbins = 10):
    '''Creates a specific colormap with the number of bins.'''
    cmap = plt.colormaps[cmapname]
    levels = MaxNLocator(nbins = nbins).tick_values(vmin, vmax)
    return(BoundaryNorm(levels, ncolors = cmap.N, clip=True))
    


def get_xylabels(n_lines, n_col, i):
    '''Return (xlabel, ylabel), two bools, which indiactes whether or not there should be a label for the x axis or y axis, based on the index i and the number of lines and cols in the plot.'''
    i_line, i_col = i//n_col, i%n_col
    xlabel, ylabel = False, False
    if i_line == (n_lines - 1):
        xlabel = True
    if i_col == 0:
        ylabel = True
    return(xlabel, ylabel)



def plot_details_axis(ax, pco, cb = True, cbarlabel = '', xlim = (-50, 60), ylim = (6000, 0), font = 15, cmapname = 'viridis', nbins = 10, title=''
                  , xlabels = True, ylabels = True, xticks = [-40+20*i for i in range(6)], yticks = [1000*i for i in range(7)]):
    '''Function to save some space for the figure which does a bunch of things depending on the arguments given.'''
    if cb:
        cbar = plt.colorbar(pco)#, ticks = [i*age_max/4 for i in range(6)])
        cbar.ax.tick_params(labelsize=font-2)
        cbar.set_label(cbarlabel, fontsize = font)
        
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    if xlabels:
        if xticks == [-40+20*i for i in range(6)]:
            ax.set_xlabel('Latitude (°N)', fontsize=font)
        elif xticks == [-80 + i * 30 for i in range(5)]:
            ax.set_xlabel('Latitude (°N)', fontsize=font)
    else: 
        ax.set_xticklabels([])
    if ylabels:
        if yticks == [1000*i for i in range(7)]:
            ax.set_yticklabels([1*i for i in range(7)],fontsize = font)
            ax.set_ylabel('Depth (km)', fontsize=font)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis='both', labelsize = font)
    ax.set_title(title, fontsize = font)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    


def add_cbar(fig, pco, x = 0.92, y = 0.1, width = 0.015, height = 0.8, fontsize = 14, label = '', ticks = []):
    cax = fig.add_axes([x, y, width, height])  # [x, y, width, height]
    cb = plt.colorbar(pco, cax=cax)
    cb.ax.tick_params(labelsize=fontsize - 1)
    cb.set_label(label, fontsize=fontsize)
    if ticks != []:
        cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=fontsize)
    return cb