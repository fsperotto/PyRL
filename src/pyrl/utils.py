# -*- coding: utf-8 -*-
"""Safe and Survival Reinforcement Learning Package.

This module implements utils.
"""

################

import pandas as pd

from IPython.display import display, HTML

import matplotlib.pyplot as plt

#from plottable import Table


################

################
# UTILS
################

#if a number is odd
def odd(n : int):
    return n%2 != 0

#if a number is even
def even(n : int):
    return n%2 == 0

################

#print a matrix using pandas dataframe and adapting presentation
def printdf(data, label_rows=None, label_cols=None, label_axis_cols=None, label_axis_rows=None, title=None, heatmap=None, precision=None, transpose=False, reverse=False, mode='html', fontsize=14):
    
    df = pd.DataFrame(data, index=label_rows, columns=label_cols)
    df = df.rename_axis(label_axis_rows)
    df = df.rename_axis(label_axis_cols, axis="columns")
    
    if transpose:
        df = df.T
    
    if reverse:
        df = df.iloc[::-1]

        
    if mode=='latex':
        print(df.to_latex())

    elif mode=='html':

        dfs = df.style
        dfs = dfs.applymap(lambda v : "color: lightgray" if v==0 else ("color: yellow" if v<0 else "color: black") )
        dfs = dfs.set_table_styles([ 
                {"selector":"th.row_heading", "props": [("border-right", "1px solid black")]},
                {"selector":"th.index_name", "props": [("border-right", "1px solid black")]},
                {"selector":"td", "props": [("padding", "2px"), ("margin", "0")]},
                {"selector":"th", "props": [("padding", "2px"), ("margin", "0")]},
            ], overwrite=False)
        dfs = dfs.set_table_styles({0:[ 
                {"selector":"td", "props": [("border-top", "1px dotted grey"), ("border-bottom", "1px dotted grey")]},
            ]}, axis=1, overwrite=False)
        dfs = dfs.set_table_styles({0:[ 
                {"selector":"td", "props": [("border-left", "1px dotted grey"), ("border-right", "1px dotted grey")]},
            ]}, overwrite=False)
        if heatmap is not None:
            dfs = dfs.background_gradient(axis=None, cmap=heatmap)
            #dfs = dfs.background_gradient(axis=1, cmap=heatmap)
            dfs = dfs.applymap(lambda v : "color: lightgray" if v==0 else "")
        if precision is not None:
            dfs = dfs.set_precision(precision)

        if title is not None:
            display(HTML('<h3>' + title + '</h3>'))
        display(dfs)

    elif mode=='text':
        print(df)

    else:
        df = df.replace(0, "")
        fig, ax = plt.subplots()
        ax.axis('off')
        if title is not None:
            fig.suptitle(title)
        #->using plottable
        #tab = Table(df)
        #->using pandas
        #table = pd.plotting.table(ax, df, loc='center', cellLoc='center')    
        #->using pyplot directly
        table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        ##table.scale(2, 2)
        ##from matplotlib import pyplot as plt
        ##import imgkit
        ##img = imgkit.from_string(dfs, False)
        ##plt.imshow(img)
        #->plot
        fig.tight_layout()
        plt.show()
