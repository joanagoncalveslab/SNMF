import math
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import sigProfilerPlotting as sigPlt
from sigProfilerPlotting import sample_portrait as sP

# For plotting
import plotly.io as plt_io
import plotly.graph_objects as go

import umap.umap_ as umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def PCA_volkova_genotoxin():
    '''Label based on Mutagen'''
    volkova_df1 = pd.read_pickle('../../CLEAN/volkova.pkl')
    volkova_df2 = pd.read_pickle('../../CLEAN/volkova2.pkl')
    df = pd.merge(volkova_df1, volkova_df2, on="Sample")
    df['Mutagen'] = df['Mutagen'].fillna(0)

    features = list(df.columns)


    y = df.loc[:, ['Mutagen','Genotype']]

    temp = df.loc[:, features[1:97]].div(df.loc[:,features[1:97]].sum(axis=1), axis=0)  # Separating out the target
    temp = pd.concat([temp, y],axis=1)
    temp.dropna(subset=features[1:97], inplace=True)
    x = temp.loc[:,temp.columns[0:96]].values
    y = temp.loc[:, 'Mutagen']
    x = StandardScaler().fit_transform(x)

    n_comp = 25
    pca = PCA(n_components=n_comp)
    pc = pca.fit_transform(x)
    num_comp = screeplot(pca)

    principal_df = pd.DataFrame(data=pc, columns=np.arange(25)+1)
    final_df = pd.concat([principal_df, temp[['Mutagen']]], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    targets = set(y)
    colors = iter(cm.rainbow(np.linspace(0, 1, len(targets))))
    for t, color in zip(targets, colors):
        # color = next(colors)
        idx = final_df['Mutagen'] == t
        ax.scatter(final_df.loc[idx, 1], final_df.loc[idx, 2],
                   c=color, s=50)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(targets, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
    plt.savefig('plots/volkova/PCA_volkova_pathway.png')
    plt.show()

    pass


def PCA_volkova_untreated():
    '''Label based on Mutagen'''
    volkova_df1 = pd.read_pickle('../../CLEAN/volkova.pkl')
    volkova_df2 = pd.read_pickle('../../CLEAN/volkova2.pkl')
    df = pd.merge(volkova_df1, volkova_df2, on="Sample")
    df['Mutagen'] = df['Mutagen'].fillna(0)

    features = list(df.columns)
    y = df.loc[:, ['Mutagen','Genotype']]
    mutagen = set(df.loc[:, ['Mutagen']].values.flatten())
    mutagen.remove(0)

    temp = df.loc[:, features[1:97]].div(df.loc[:,features[1:97]].sum(axis=1), axis=0)  # Separating out the target
    temp = pd.concat([temp, y],axis=1)
    temp.dropna(subset=features[1:97], inplace=True)
    temp = temp[~temp['Mutagen'].isin(mutagen)]

    #PATHWAY
    xls = pd.ExcelFile('../../data/volkova2021/KO_pathway.xlsx')
    pathways = pd.read_excel(xls, 'Blad1', header=0)
    temp = pd.merge(temp, pathways, on='Genotype')

    x = temp.loc[:,temp.columns[0:96]].values
    y = temp.loc[:, 'Pathway']
    x = StandardScaler().fit_transform(x)

    n_comp = 25
    pca = PCA(n_components=n_comp)
    pc = pca.fit_transform(x)
    num_comp = screeplot(pca)

    principal_df = pd.DataFrame(data=pc, columns=np.arange(25)+1)
    final_df = pd.concat([principal_df, y], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    targets = set(y)
    colors = iter(cm.rainbow(np.linspace(0, 1, len(targets))))
    for t, color in zip(targets, colors):
        # color = next(colors)
        idx = final_df['Pathway'] == t
        ax.scatter(final_df.loc[idx, 1], final_df.loc[idx, 2],
                   c=color, s=50)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(targets, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
    plt.savefig('plots/volkova/volkova_pathway_untreated.png')
    plt.show()

    pass

def screeplot(pca,title):
    plt.figure()
    var = pca.explained_variance_ratio_
    cumsum = np.cumsum(var)
    plt.plot(np.arange(pca.n_components)+1, var, 'ro-')
    plt.plot(np.arange(var.size) + 1, cumsum)
    plt.title('Scree Plot')
    plt.xlabel('PC')
    plt.ylabel('Proportion of Variance explained')
    plt.show()
    plt.savefig('plots/volkova/{0}_{1}_{2}_{3}_Screeplot.png'.format(*title))

    return 2


def UMAP_volkova():
    # [ DATASET ; LABELED ; FILTER ; COMPARE to ]
    title = ['volkova','pathway','untreated','MMR']

    volkova_df1 = pd.read_pickle('../../CLEAN/volkova.pkl')
    volkova_df2 = pd.read_pickle('../../CLEAN/volkova2.pkl')
    df = pd.merge(volkova_df1, volkova_df2, on="Sample")
    df['Mutagen'] = df['Mutagen'].fillna(0)
    features = list(df.columns)
    df['Count'] = df.loc[:,features[1:97]].sum(axis=1)

    y = df.loc[:, ['Mutagen','Genotype','Generation']]

    # # Filter Gene (IF specific gene in pathway
    # gene = set(df.loc[:, ['Genotype']].values.flatten())
    # gene.difference_update(['N2','brc-1','rip-1','rfs-1'])
    #
    # # Filter Generation
    # generation = set(df.loc[:, ['Generation']].values.flatten())
    # generation.difference_update([20,40])
    #
    # # Filter Mutagen -> only untreated
    mutagen = set(df.loc[:, ['Mutagen']].values.flatten())
    mutagen.remove(0)

    temp = df.loc[:, features[1:97]].div(df.loc[:,features[1:97]].sum(axis=1), axis=0)
    temp = pd.concat([temp, y],axis=1)
    temp.dropna(subset=features[1:97], inplace=True)
    temp = temp[~temp['Mutagen'].isin(mutagen)]
    # temp = temp[~temp['Genotype'].isin(gene)]
    # temp = temp[~temp['Generation'].isin(generation)]

    #PATHWAY
    xls = pd.ExcelFile('../../data/volkova2021/KO_pathway.xlsx')
    pathways = pd.read_excel(xls, 'Blad1', header=0)
    temp = pd.merge(temp, pathways, on='Genotype')
    pathway = set(temp.loc[:, ['Pathway']].values.flatten())
    pathway.difference_update(['Control','MMR'])
    temp = temp[~temp['Pathway'].isin(pathway)]


    x = temp.loc[:,temp.columns[0:96]].values
    y = temp.loc[:, 'Pathway']
    y = y.reset_index(drop=True)
    x = StandardScaler().fit_transform(x)

    #UMAP
    reducer = umap.UMAP(random_state=42, n_components=2)
    embedding = reducer.fit_transform(x)

    final_df = pd.DataFrame(data=embedding, columns=np.arange(2) + 1)
    final_df = pd.concat([final_df, y], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_title('UMAP volkova all (untreated samples)')
    ax.set_title('UMAP volkova all')

    targets = set(y)
    colors = iter(cm.rainbow(np.linspace(0, 1, len(targets))))
    for t, color in zip(targets, colors):
        # color = next(colors)
        idx = final_df['Pathway'] == t
        ax.scatter(final_df.loc[idx, 1], final_df.loc[idx, 2],
                   c=color, s=50)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(targets, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
    plt.savefig('plots/volkova/{0}_{1}_{2}_{3}_UMAP.png'.format(*title))
    plt.show()

    # PCA
    n_comp = 10
    pca = PCA(n_components=n_comp)
    pc = pca.fit_transform(x)
    num_comp = screeplot(pca,title)

    final_df_PCA = pd.DataFrame(data=pc, columns=np.arange(n_comp)+1)
    final_df_PCA = pd.concat([final_df_PCA, y], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_title('PCA volkova DSBR/HR vs WT (untreated samples)')
    ax.set_title('PCA volkova MMR')
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    targets = set(y)
    colors = iter(cm.rainbow(np.linspace(0, 1, len(targets))))
    for t, color in zip(targets, colors):
        # color = next(colors)
        idx = final_df_PCA['Pathway'] == t
        ax.scatter(final_df_PCA.loc[idx, 1], final_df_PCA.loc[idx, 2],
                   c=color, s=50)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(targets, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
    plt.savefig('plots/volkova/{0}_{1}_{2}_{3}_PCA.png'.format(*title))
    plt.show()

    pass

# PCA_volkova_untreated()
UMAP_volkova()