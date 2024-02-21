import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Patch
import matplotlib.image as mpimg
from matplotlib import rcParams

import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import sigProfilerPlotting as sigPlt
import umap.umap_ as umap
import os
os.chdir("C:/Users/sande/PycharmProjects/MEP_data")


def bootstrap_perSample(data, data_SD, L ,sample_count=False):
    np.random.seed(0)

    feat = data.reset_index().columns
    genes = data.index
    samples_boot = []

    for i in genes:
        for l in range(L):
            samples_boot.append(i + '_' + str(l))
    data_boot = pd.DataFrame(0, index=samples_boot, columns=feat)

    for index, row in data.iterrows():
        gene = index
        N = round(int(row['Count']))

        for l in range(L):
            if sample_count:
                # Sample from Normal distribution --> Poisson or Negative Binomial?
                N = round(int(row['Count']) + np.random.randn() * int(data_SD.loc[index, 'Count']))

            sample = gene + '_' + str(l)
            data_boot.loc[sample, 'Gene_KO'] = row['Gene_KO']
            data_boot.loc[sample, 'Protein_KO'] = row['Protein_KO']
            data_boot.loc[sample, 'Subpathway_KO'] = row['Subpathway_KO']
            data_boot.loc[sample, 'label'] = row['label']

            data_boot.loc[sample, 'Count'] = N

            Pr = row[feat[1:97]]
            Pr = Pr.tolist()

            muts = np.random.choice(feat[1:97], N, replace=True, p=Pr)
            for m in muts:
                # Add bootstrapped/sampled mutations to count matrix
                data_boot.loc[sample, m] += 1

    data_boot[feat[1:97]] = data_boot[feat[1:97]].div(data_boot['Count'].values, axis=0)

    data_comb = pd.concat([data_boot, data])

    for index, row in data_comb.iterrows():
        if len(index.split('_')) == 4:
            data_comb.loc[index, 'Gene_KO_paired'] = row['Gene_KO'] + '_' + 'boot'
            data_comb.loc[index, 'Subpathway_KO_paired'] = row['Subpathway_KO'] + '_' + 'boot'
        else:
            data_comb.loc[index, 'Gene_KO_paired'] = row['Gene_KO'] + '_' + 'real'
            data_comb.loc[index, 'Subpathway_KO_paired'] = row['Subpathway_KO'] + '_' + 'real'

    return data_comb



def bootstrap_equal(data, L):
    '''
    :param data:
    :param L: wanted total samples per class
    :return: Combined Real and Bootstrap data (combined number of samples per class is L)
    '''
    np.random.seed(0)
    feat = data.columns

    # For loop over classes
    # Oversample to L -> randomly select (L - #real samples) samples with replacement -> bootstrap selected samples
    # Concatenate Real and Bootstrapped Samples

    classes = data['label'].unique()
    new_samples = []

    for c in classes:
        current = data.loc[data['label'] == c]
        n_real = current.shape[0]
        n_boot = L - n_real

        boot_idx = np.random.choice(n_real, replace = True, size = n_boot)
        to_boot = current.iloc[boot_idx]
        to_boot.dropna(inplace = True)

        x = 1
        for index, row in to_boot.iterrows():
            N = round(int(row['Count']))
            Pr = row[feat[0:96]]

            profile = [0]*96
            muts = np.random.choice(96, N, replace=True, p=Pr)
            frac = 1/N
            for m in muts:
                # Add bootstrapped/sampled mutations to count matrix
                profile[m] += frac

            profile.append(row['Gene_KO'])
            profile.append(row['Count'])
            profile.append(row['Protein_KO'])
            profile.append(row['Subpathway_KO'])
            profile.append(row['label'])

            new_samples.append(profile)
            x += 1

    data_boot = pd.DataFrame(new_samples)
    data_boot.columns = list(feat)

    data_comb = pd.concat([data_boot, data])

    for index, row in data_comb.iterrows():
        if isinstance(index, int):
            data_comb.loc[index, 'boot'] = True
            data_comb.loc[index, 'Gene_KO_paired'] = row['Gene_KO'] + '_' + 'boot'
            data_comb.loc[index, 'Subpathway_KO_paired'] = row['Subpathway_KO'] + '_' + 'boot'
        else:
            data_comb.loc[index, 'boot'] = False
            data_comb.loc[index, 'Gene_KO_paired'] = row['Gene_KO'] + '_' + 'real'
            data_comb.loc[index, 'Subpathway_KO_paired'] = row['Subpathway_KO'] + '_' + 'real'

    return data_comb


def bootstrap_uncor(data, L):
    '''
    :param data:
    :param L: wanted total samples per class
    :return: Combined Real and Bootstrap data (combined number of samples per class is L)
    '''
    np.random.seed(0)
    feat = data.columns

    # For loop over classes
    # Oversample to L -> randomly select (L - #real samples) samples with replacement -> bootstrap selected samples
    # Concatenate Real and Bootstrapped Samples

    classes = data['label'].unique()
    new_samples = []

    for c in classes:
        current = data.loc[data['label'] == c]
        n_real = current.shape[0]
        n_boot = L - n_real

        boot_idx = np.random.choice(n_real, replace = True, size = n_boot)
        to_boot = current.iloc[boot_idx]
        to_boot.dropna(inplace = True)

        x = 1
        for index, row in to_boot.iterrows():
            N = round(int(row['Count']))
            Pr = row[feat[0:96]]

            profile = [0]*96
            muts = np.random.choice(96, N, replace=True, p=Pr)
            frac = 1/N
            for m in muts:
                # Add bootstrapped/sampled mutations to count matrix
                profile[m] += frac

            profile.append(row['Gene_KO'])
            profile.append(row['Count'])
            profile.append(row['Protein_KO'])
            profile.append(row['Subpathway_KO'])
            profile.append(row['label'])

            new_samples.append(profile)
            x += 1

    data_boot = pd.DataFrame(new_samples)
    data_boot.columns = list(feat)

    data_comb = pd.concat([data_boot, data])

    for index, row in data_comb.iterrows():
        if isinstance(index, int):
            data_comb.loc[index, 'boot'] = True
            data_comb.loc[index, 'Gene_KO_paired'] = row['Gene_KO'] + '_' + 'boot'
            data_comb.loc[index, 'Subpathway_KO_paired'] = row['Subpathway_KO'] + '_' + 'boot'
        else:
            data_comb.loc[index, 'boot'] = False
            data_comb.loc[index, 'Gene_KO_paired'] = row['Gene_KO'] + '_' + 'real'
            data_comb.loc[index, 'Subpathway_KO_paired'] = row['Subpathway_KO'] + '_' + 'real'

    return data_comb


# zou = pd.read_pickle("data/zou2021/zou_96SBS_filtered.pkl")
# zou_u = pd.read_pickle("data/zou2021/zou_96SBS_filtered_mean.pkl")
# zou_SD = pd.read_pickle("data/zou2021/zou_96SBS_filtered_SD.pkl")
# zou_comb = bootstrap_perSample(zou, zou_SD, 5)
