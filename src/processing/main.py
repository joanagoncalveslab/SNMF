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

def mutprof(data,project):
    # sP.samplePortrait()
    output_path = "../../plots/"
    sigPlt.plotSBS(data, project = project, output_path="../../plots/", plot_type="96", percentage=True)
    return


volkova_df1 = pd.read_pickle('../../CLEAN/volkova.pkl')
volkova_df2 = pd.read_pickle('../../CLEAN/volkova2.pkl')

def volkova_mmr():
    volkova_df1 = pd.read_pickle('../../CLEAN/volkova.pkl')
    volkova_df2 = pd.read_pickle('../../CLEAN/volkova2.pkl')
    mmr = []
    result = [0]*96
    for i in range(2718):
        if volkova_df2.iat[i,1] == 'mlh-1':
            mmr.append(i)
            for j in range(1, 97):
                result[j-1] += volkova_df1.iat[i,j]
    filename = '../../CLEAN/volkova_mmr2.all'
    f = open(filename, 'w')
    idx = volkova_df1.columns.tolist()
    l = []
    l.append('MutationType\tCount')
    for i in range(96):
        l.append(idx[i + 1] + '\t' + str(int(result[i])))
    f.write('\n'.join(l))
    f.close()
    mutprof(filename, 'mmr')


def volkova_sum():
    volkova_df1 = pd.read_pickle('../../CLEAN/volkova.pkl')

    volkova_df1.loc['total'] = volkova_df1.select_dtypes(pd.np.number).sum()
    total_profile = volkova_df1.iloc[-1]

    f = open('../../CLEAN/sum2.all', 'w')
    idx = total_profile.index.tolist()
    l = []
    l.append('MutationType\tCount')

    for i in range(96):
        l.append(idx[i+1]  + '\t' + str(int(total_profile[i+1])) )
    f.write('\n'.join(l) )

    mutprof('sum3.txt', 'sum3')

def volkova_boxplot():
    volkova_df1 = pd.read_pickle('../../CLEAN/volkova.pkl')
    volkova_col = list(volkova_df1.columns)
    volkova_df1.boxplot(column=volkova_col[1:96])
    plt.show()

def volkova_boxplot_all():
    volkova_df1 = pd.read_pickle('../../CLEAN/volkova.pkl')
    volkova_df2 = pd.read_pickle('../../CLEAN/volkova2.pkl')
    mmr = []
    volkova_col = list(volkova_df1.columns)
    df = pd.DataFrame(columns=volkova_col, index=range(1084))

    for i in range(2718):
        for j in range(1, 97):
            df.iat[int(volkova_df1.iat[i,0][2:6])-1,j] += volkova_df1.iat[i,j]


    df.boxplot()
    plt.show()

def PCA_volkova():
    volkova_df1 = pd.read_pickle('../../CLEAN/volkova.pkl')
    volkova_df2 = pd.read_pickle('../../CLEAN/volkova2.pkl')
    df = pd.merge(volkova_df1, volkova_df2, on="Sample")

    # Mutation types
    features = list(volkova_df1.columns)
    # Separating out the features
    x = df.loc[:, features[1:97]].values  # Separating out the target
    y = df.loc[:, 'Genotype'].values  # Standardizing the features
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    pc = pca.fit_transform(x)
    principal_df = pd.DataFrame(data= pc, columns= ['PC1', 'PC2'])
    final_df = pd.concat([principal_df, df[['Genotype']]], axis =1)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    targets = set(y)
    colors = iter(cm.rainbow(np.linspace(0,1,len(targets))))
    for t, color in zip(targets,colors):
        # color = next(colors)
        idx = final_df['Genotype'] == t
        ax.scatter(final_df.loc[idx,'PC1'], final_df.loc[idx,'PC2'],
                    c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.savefig('pca_volkova_all.svg')
    plt.show()

def PCA_volkova_join():
    volkova_df1 = pd.read_pickle('../../CLEAN/volkova.pkl')
    volkova_df2 = pd.read_pickle('../../CLEAN/volkova2.pkl')
    df = pd.merge(volkova_df1, volkova_df2, on="Sample")

    features = list(df.columns)
    df2 = df.loc[:, features[1:121]].groupby(['Genotype'], as_index = False).agg('sum')
    # Mutation types
    # Separating out the features
    test = df2.loc[:, features[1:97]].div(df2.sum(axis=1), axis=0)
    plt.imshow(test, cmap='hot', interpolation='nearest')
    plt.show()
    x = df2.loc[:, features[1:97]].div(df2.sum(axis=1), axis=0).values  # Separating out the target
    y = set(df.loc[:, 'Genotype'].values)  # Standardizing the features

    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    pc = pca.fit_transform(x)
    principal_df = pd.DataFrame(data= pc, columns= ['PC1', 'PC2'])
    final_df = pd.concat([principal_df, df2[['Genotype']]], axis =1)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    targets = set(y)
    colors = iter(cm.rainbow(np.linspace(0,1,len(targets))))
    for t, color in zip(targets,colors):
        # color = next(colors)
        idx = final_df['Genotype'] == t
        ax.scatter(final_df.loc[idx,'PC1'], final_df.loc[idx,'PC2'],
                    c=color, s=50)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(targets, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
    plt.savefig('pca_volkova_pathway.svg')
    plt.show()

def zou2018_profiles():
    # zou2018_u = pd.read_pickle('data/zou2018/zou2018_child_sum.pkl')

    # zou2018_u.transpose().to_csv('test2.txt', sep='\t', index=True)
    mutprof('test2.txt', 'zou2018_2')

    pass



def zou2018_cossim():
    zou2018_u = pd.read_pickle('../../data/zou2018/zou2018_child_sum.pkl')
    # zou_2018_parent = pd.read_pickle('data/zou2018/zou2018_parent.pkl')
    test = zou2018_u.loc[:, :].div(zou2018_u.sum(axis=1), axis=0)
    data = test.values
    similarity = np.zeros((9,9))
    for i in range(9):
        for j in range(i+1):
            similarity[i,j] = cossim(data[i,:], data[j,:])
            similarity[j, i] = similarity[i,j]
    genes = zou2018_u.index.values
    ax = sns.heatmap(similarity, annot=True, xticklabels = genes, yticklabels= genes)
    plt.savefig('data/zou2018/zou2018_heatmap_genes.svg')
    pass


def cosim_matrix(D):
    ''' cosine similarity matrix of rows in dataset'''
    # for i in D
    pass

def cossim(A, B):
    '''return cosine similarity between 2 vectors'''
    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

def cosmic_profile():
    mutprof('data/COSMIC_v3.2_SBS_GRCh37.txt', 'COSMIC')

def cosmic_cossim():
    zou2018_u = pd.read_pickle('../../data/zou2018/zou2018_child_sum.pkl')
    # zou_2018_parent = pd.read_pickle('data/zou2018/zou2018_parent.pkl')
    test = zou2018_u.loc[:, :].div(zou2018_u.sum(axis=1), axis=0)
    data = test.values

    cosmic_temp = pd.read_csv('../../data/COSMIC_v3.2_SBS_GRCh37.txt', delimiter='\t', index_col='Type')
    cosmic = cosmic_temp.values
    similarity = np.zeros((9,78))
    for i in range(9):
        for j in range(78):
            similarity[i,j] = cossim(data[i,:], cosmic[:,j])
            # similarity[j, i] = similarity[i,j]
    genes = zou2018_u.index.values
    SBS = cosmic_temp.columns.values
    # ax = sns.heatmap(similarity, annot=True, xticklabels = genes, yticklabels= SBS)
    ax = sns.heatmap(similarity, xticklabels = SBS, yticklabels= genes)
    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % 5 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.savefig('data/zou2018/zou2018_heatmap_COSMIC.svg')
    pass



def volkova_mutcount():
    volkova_df1 = pd.read_pickle('../../CLEAN/volkova.pkl')
    volkova_df2 = pd.read_pickle('../../CLEAN/volkova2.pkl')
    df = pd.merge(volkova_df1, volkova_df2, on="Sample")

    features = list(df.columns)
    df2 = df.loc[:, features[1:121]].groupby(['Genotype'], as_index = False).agg('sum')
    # Mutation types
    # Separating out the features
    counts_sample = df.loc[:, features[1:97]].sum(axis=1)
    counts_gene = df2.loc[:, features[1:97]].sum(axis=1)
    fig, ax = plt.subplots(2,1)
    sns.histplot(ax = ax[0], data = counts_sample)
    ax[0].set_title('number of mutations per sample')
    sns.histplot(ax=ax[1], data=counts_gene)
    ax[1].set_title('number of mutations per gene KO (sum over samples)')
    plt.subplots_adjust(hspace=0.9)
    plt.savefig('data/volkova2021/volkova2021_counts.svg')
    pass

def volkova_cossim():
    volkova_df1 = pd.read_pickle('../../CLEAN/volkova.pkl')
    volkova_df2 = pd.read_pickle('../../CLEAN/volkova2.pkl')
    df = pd.merge(volkova_df1, volkova_df2, on="Sample")
    features = list(df.columns)
    df2 = df.loc[:, features[1:121]].groupby(['Genotype'], as_index = False).agg('sum')
    test = df2.loc[:, features[1:97]].div(df2.sum(axis=1), axis=0)

    data = test.values
    similarity = np.zeros((53,53))
    for i in range(53):
        for j in range(i+1):
            similarity[i,j] = cossim(data[i,:], data[j,:])
            similarity[j, i] = similarity[i,j]
    ax = sns.heatmap(similarity)
    plt.savefig('data/volkova2021/volkova2021_cossim')
    pass

def load_zou2021():
    xls = pd.ExcelFile('../../data/zou2021/Supplementary_tables.xlsx')
    # S3 = Mutational profiles per Sample  (96 x 173)
    profiles = pd.read_excel(xls, 'TableS3', index_col= 0,header= 0)
    # S1 = Gene - Pathway mapping   (43 KOs
    KO_pathway = pd.read_excel(xls, 'TableS1', index_col= 0, header= 0)
    # zou2021.rename(columns = {"Unnamed: 0":"Sample"}, inplace=True)

    # To look-up tables Gene <-> Pathway
    pathways = {}
    genes = {}
    for index, row in KO_pathway.iterrows():
        p = row['Subpathway_KO'].split('/')
        for i in p:
            if i in pathways:
                pathways[i].append(index)
            else:
                pathways[i] = [index]
            if index in genes:
                genes[index].append(i)
            else:
                genes[index] = [i]

    features = list(profiles.index)

    profiles = profiles.drop(columns=['Mutation'])
    samples = list(profiles.columns)
    profiles = profiles.loc[:,:].div(profiles.sum(axis=0), axis=1)
    profiles = profiles.transpose()
    gene_KO = []
    for s in samples:
        gene_KO.append(s.split('_')[0])
    profiles['KO'] = gene_KO
    joined = pd.merge(left=profiles, right=KO_pathway, how= 'left', left_on='KO', right_index = True)
    # plt.imshow(profiles, cmap='hot', interpolation='nearest')
    # plt.show()
    x = profiles.loc[:,features].values  # Separating out the target
    y = profiles.loc[:, 'KO'].values  # Standardizing the features

    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    pc = pca.fit_transform(x)
    principal_df = pd.DataFrame(data= pc, columns= ['PC1', 'PC2'])
    final_df = pd.concat([principal_df, profiles[['KO']].reset_index(drop=True)], axis =1)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    targets = set(y)
    colors = iter(cm.rainbow(np.linspace(0,1,len(targets))))
    for t in targets:
        c = next(colors)
        idx = final_df['KO'] == t
        ax.scatter(final_df.loc[idx,'PC1'], final_df.loc[idx,'PC2'],
                    c=c, s=50)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(targets, loc='center left', bbox_to_anchor=(1, 0.5), fontsize= 'x-small')
    ax.grid()
    plt.show()
    plt.savefig('data/zou2021/pca_zou2021_KO.png')


    pass

def load_zou2021_pathway():
    xls = pd.ExcelFile('../../data/zou2021/Supplementary_tables.xlsx')
    # S3 = Mutational profiles per Sample  (96 x 173)
    profiles = pd.read_excel(xls, 'TableS3', index_col= 0,header= 0)
    # S1 = Gene - Pathway mapping   (43 KOs
    KO_pathway = pd.read_excel(xls, 'TableS1', index_col= 0, header= 0)
    # zou2021.rename(columns = {"Unnamed: 0":"Sample"}, inplace=True)

    # To look-up tables Gene <-> Pathway
    pathways = {}
    genes = {}
    for index, row in KO_pathway.iterrows():
        p = row['Subpathway_KO'].split('/')
        for i in p:
            if i in pathways:
                pathways[i].append(index)
            else:
                pathways[i] = [index]
            if index in genes:
                genes[index].append(i)
            else:
                genes[index] = [i]

    features = list(profiles.index)
    profiles.co

    profiles = profiles.drop(columns=['Mutation'])
    samples = list(profiles.columns)
    profiles = profiles.loc[:,:].div(profiles.sum(axis=0), axis=1)
    profiles = profiles.transpose()
    gene_KO = []
    for s in samples:
        gene_KO.append(s.split('_')[0])
    profiles['KO'] = gene_KO
    joined = pd.merge(left=profiles, right=KO_pathway, how= 'left', left_on='KO', right_index = True)
    # plt.imshow(profiles, cmap='hot', interpolation='nearest')
    # plt.show()
    x = profiles.loc[:,features].values  # Separating out the target
    y = joined.loc[:, 'Subpathway_KO'].values  # Standardizing the features

    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    pc = pca.fit_transform(x)
    principal_df = pd.DataFrame(data= pc, columns= ['PC1', 'PC2'])
    final_df = pd.concat([principal_df, joined[['Subpathway_KO']].reset_index(drop=True)], axis =1)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.title.set_text('PCA Zou (2021); 173 samples; 43 KOs; 13 Pathways')
    targets = set(y)
    colors = iter(cm.rainbow(np.linspace(0,1,len(targets))))
    for t in targets:
        c = next(colors)
        idx = final_df['Subpathway_KO'] == t
        ax.scatter(final_df.loc[idx,'PC1'], final_df.loc[idx,'PC2'],
                    c=c, s=50)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(targets, loc='center left', bbox_to_anchor=(1, 0.5), fontsize= 'x-small')
    ax.grid()
    plt.show()
    plt.savefig('data/zou2021/pca_zou2021_Pathway.png')

def PCA_volkova_genes():
    volkova_df1 = pd.read_pickle('../../CLEAN/volkova.pkl')
    volkova_df2 = pd.read_pickle('../../CLEAN/volkova2.pkl')
    df = pd.merge(volkova_df1, volkova_df2, on="Sample")

    features = list(df.columns)
    df2 = df.loc[:, features[1:121]].groupby(['Genotype'], as_index=False).agg('sum')
    # Mutation types
    # Separating out the features
    test = df2.loc[:, features[1:97]].div(df2.sum(axis=1), axis=0)
    plt.imshow(test, cmap='hot', interpolation='nearest')
    plt.show()
    temp = df.loc[:, features[1:97]].div(df.loc[:,features[1:97]].sum(axis=1), axis=0)  # Separating out the target
    temp.dropna(subset=features[1:97], inplace=True)
    x = temp.values
    y = df.loc[:, 'Genotype'].values  # Standardizing the features

    x = StandardScaler().fit_transform(x)


    pca = PCA(n_components=2)
    pc = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=pc, columns=['PC1', 'PC2'])
    final_df = pd.concat([principal_df, df[['Genotype']]], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    targets = set(y)
    colors = iter(cm.rainbow(np.linspace(0, 1, len(targets))))
    for t, color in zip(targets, colors):
        # color = next(colors)
        idx = final_df['Genotype'] == t
        ax.scatter(final_df.loc[idx, 'PC1'], final_df.loc[idx, 'PC2'],
                   c=color, s=50)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(targets, loc='center left', bbox_to_anchor=(1, 0.5), fontsize= 'xx-small')
    ax.grid()
    plt.savefig('data/volkova2021/pca_volkova_KOs.png')
    plt.show()

    pass

def PCA_volkova_pathway():
    volkova_df1 = pd.read_pickle('../../CLEAN/volkova.pkl')
    volkova_df2 = pd.read_pickle('../../CLEAN/volkova2.pkl')
    xls = pd.ExcelFile('../../data/volkova2021/KO_pathway.xlsx')
    pathways = pd.read_excel(xls, 'Blad1', header=0)
    df = pd.merge(volkova_df1, volkova_df2, on="Sample")
    df_path = pd.merge(df, pathways, on='Genotype')

    features = list(df.columns)
    df2 = df.loc[:, features[1:121]].groupby(['Genotype'], as_index=False).agg('sum')
    # Mutation types
    # Separating out the features
    test = df2.loc[:, features[1:97]].div(df2.sum(axis=1), axis=0)
    plt.imshow(test, cmap='hot', interpolation='nearest')
    plt.show()
    temp = df.loc[:, features[1:97]].div(df.loc[:,features[1:97]].sum(axis=1), axis=0)  # Separating out the target
    temp.dropna(subset=features[1:97], inplace=True)
    x = temp.values
    y = df_path.loc[:, 'Unnamed: 2'].values  # Standardizing the features

    x = StandardScaler().fit_transform(x)


    pca = PCA(n_components=2)
    pc = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=pc, columns=['PC1', 'PC2'])
    final_df = pd.concat([principal_df, df_path[['Pathway']]], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    targets = set(y)
    colors = iter(cm.rainbow(np.linspace(0, 1, len(targets))))
    for t, color in zip(targets, colors):
        # color = next(colors)
        idx = final_df['Pathway'] == t
        ax.scatter(final_df.loc[idx, 'PC1'], final_df.loc[idx, 'PC2'],
                   c=color, s=50)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(targets, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
    plt.savefig('data/volkova2021/pca_volkova_pathway.png')
    plt.show()

    pass

def PCA_volkova_zou_pathway():
    volkova_df1 = pd.read_pickle('../../CLEAN/volkova.pkl')
    volkova_df2 = pd.read_pickle('../../CLEAN/volkova2.pkl')
    xls = pd.ExcelFile('../../data/volkova2021/KO_pathway.xlsx')
    pathways = pd.read_excel(xls, 'Blad1', header=0)
    df = pd.merge(volkova_df1, volkova_df2, on="Sample")
    df_path = pd.merge(df, pathways, on='Genotype')

    features = list(df.columns)
    df2 = df.loc[:, features[1:121]].groupby(['Genotype'], as_index=False).agg('sum')
    # Mutation types
    # Separating out the features
    test = df2.loc[:, features[1:97]].div(df2.sum(axis=1), axis=0)
    plt.imshow(test, cmap='hot', interpolation='nearest')
    plt.show()
    temp = df.loc[:, features[1:97]].div(df.loc[:,features[1:97]].sum(axis=1), axis=0)  # Separating out the target
    temp.dropna(subset=features[1:97], inplace=True)
    x = temp.values
    y = df_path.loc[:, 'Pathway'].values  # Standardizing the features

    x = StandardScaler().fit_transform(x)


    pca = PCA(n_components=2)
    pc = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=pc, columns=['PC1', 'PC2'])
    final_df = pd.concat([principal_df, df_path[['Pathway']]], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    targets = set(y)
    colors = iter(cm.rainbow(np.linspace(0, 1, len(targets))))
    for t, color in zip(targets, colors):
        # color = next(colors)
        idx = final_df['Unnamed: 2'] == t
        ax.scatter(final_df.loc[idx, 'PC1'], final_df.loc[idx, 'PC2'],
                   c=color, s=50)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(targets, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
    plt.savefig('data/volkova2021/pca_volkova_pathway.png')
    plt.show()

    pass

def PCA_volkova_pathway():
    volkova_df1 = pd.read_pickle('../../CLEAN/volkova.pkl')
    volkova_df2 = pd.read_pickle('../../CLEAN/volkova2.pkl')
    xls1 = pd.ExcelFile('../../data/volkova2021/KO_pathway.xlsx')
    pathways = pd.read_excel(xls1, 'Blad1', header=0)
    df = pd.merge(volkova_df1, volkova_df2, on="Sample")
    df_path = pd.merge(df, pathways, on='Genotype')

    xls2 = pd.ExcelFile('../../data/zou2021/Supplementary_tables.xlsx')
    # S3 = Mutational profiles per Sample  (96 x 173)
    profiles = pd.read_excel(xls2, 'TableS3', index_col=0, header=0)
    # S1 = Gene - Pathway mapping   (43 KOs
    KO_pathway = pd.read_excel(xls2, 'TableS1', index_col=0, header=0)
    pathways = {}
    genes = {}
    for index, row in KO_pathway.iterrows():
        p = row['Subpathway_KO'].split('/')
        for i in p:
            if i in pathways:
                pathways[i].append(index)
            else:
                pathways[i] = [index]
            if index in genes:
                genes[index].append(i)
            else:
                genes[index] = [i]
    remove = set()
    for index, row in df_path.iterrows():
        if row['Mutagen'] == row['Mutagen']:
            remove.add(index)
        if (row['Pathway'] != 'DSBR/HR' and row['Pathway'] != 'Control'):
            remove.add(index)
    volkova_MMR = df_path.drop(list(remove), axis=0)


    profiles = profiles.drop(columns=['Mutation'])
    samples = list(profiles.columns)
    profiles = profiles.loc[:,:].div(profiles.sum(axis=0), axis=1)
    profiles = profiles.transpose()
    gene_KO = []
    for s in samples:
        gene_KO.append(s.split('_')[0])
    profiles['KO'] = gene_KO
    joined = pd.merge(left=profiles, right=KO_pathway, how= 'left', left_on='KO', right_index = True)
    remove = set()
    for index, row in joined.iterrows():
        if 'EXO1' not in row['KO'].split('/') and 'Control' not in row['Subpathway_KO'].split('/'):
            remove.add(index)
    zou_MMR = joined.drop(list(remove), axis=0)
    zou_temp = zou_MMR

    features = list(df.columns)
    volkova_features = features[1:97].copy()
    volkova_features.append('Pathway')
    zou_features = features[1:97].copy()
    zou_features.append('Subpathway_KO')
    volkova_MMR = volkova_MMR[volkova_features]
    zou_MMR = zou_MMR[zou_features]
    zou_labels = zou_MMR['Subpathway_KO']

    volkova_labels = volkova_MMR['Pathway']
    volkova_MMR = volkova_MMR.loc[:, features[1:97]].div(volkova_MMR.sum(axis=1), axis=0)
    volkova_MMR = volkova_MMR.assign(labels=volkova_labels)
    volkova_MMR.dropna(subset=features[1:97], inplace=True)
    volkova_labels = volkova_MMR['labels']
    all_data = pd.concat([volkova_MMR,zou_MMR])
    x = all_data.loc[:,features[1:97]].values
    y = pd.concat([volkova_labels,zou_labels])

    x = StandardScaler().fit_transform(x)


    n_comp = 25
    pca = PCA(n_components=n_comp)
    pc = pca.fit_transform(x)
    num_comp = screeplot(pca)

    principal_df = pd.DataFrame(data=pc, columns=np.arange(25)+1)
    final_df = pd.concat([principal_df, y.reset_index()[0]], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)

    for index, row in final_df.iterrows():
        if index <= 91:
            final_df.loc[index,0] = final_df.loc[index,0] + " - Volkova"
        else:
            final_df.loc[index, 0] = final_df.loc[index, 0] + " - Zou"
    y = final_df[0]
    targets = set(y)
    colors = iter(cm.rainbow(np.linspace(0, 1, len(targets))))
    for t, color in zip(targets, colors):
        # color = next(colors)
        idx = final_df[0] == t
        ax.scatter(final_df.loc[idx, 1], final_df.loc[idx, 2],
                   c=color, s=50)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(targets, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
    plt.savefig('plots/multi/pca_DSBR-HR_zou_volkova_3.png')
    plt.show()

    pass

def screeplot(pca):
    var = pca.explained_variance_ratio_
    cumsum = np.cumsum(var)
    plt.plot(np.arange(pca.n_components)+1, var, 'ro-')
    plt.plot(np.arange(var.size) + 1, cumsum)
    plt.title('Scree Plot')
    plt.xlabel('PC')
    plt.ylabel('Proportion of Variance explained')
    plt.show()
    plt.savefig('plots/multi/pca_MMR_zou_volkova_screeplot_{}.png'.format(pca.n_components))

    return 2


# volkova_sum()
# volkova_mmr()
# volkova_boxplot_all()
# PCA_volkova_join()
# PCA_volkova_genes()
# zou2018_profiles()
# zou2018_cossim()
# cosmic_cossim()
# volkova_mutcount()
# volkova_cossim()
# load_zou2021_pathway()
PCA_volkova_pathway()

pass