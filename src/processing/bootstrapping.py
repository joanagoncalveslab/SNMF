import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Patch
import matplotlib.image as mpimg
from matplotlib import rcParams
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors


import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import sigProfilerPlotting as sigPlt
import umap.umap_ as umap
import os
# os.chdir("C:/Users/sande/PycharmProjects/MEP_data")


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


def bootstrap_dirichlet(data, L=100, dirichlet_strength=50, epsilon=1e-3, suffix='train', class_col='label'):
    """
    Generate bootstrap samples from real data using both multinomial and dirichlet sampling.

    Args:
        data (pd.DataFrame): Input data with mutation profiles and metadata.
        L (int): Target number of samples per class.
        dirichlet_strength (float): Concentration parameter for the Dirichlet distribution.
        epsilon (float): Smoothing term to avoid zero probabilities.
        suffix (str): Identifier for where this data came from (e.g., 'train1').
        class_col (str): Column name to stratify bootstrapping on (e.g., 'label' or 'Gene_KO').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Multinomial-bootstrapped and Dirichlet-bootstrapped dataframes.
    """
    np.random.seed(0)

    mut_cols = data.columns[:96]
    meta_cols = data.columns[96:]
    XbootM, XbootD = [], []

    for c in data[class_col].unique():
        current = data[data[class_col] == c]
        n_real = current.shape[0]
        n_boot = L - n_real
        boot_idx = np.random.choice(n_real, replace=True, size=n_boot)
        to_boot = current.iloc[boot_idx].copy().dropna()

        for _, row in to_boot.iterrows():
            N = round(int(row["Count"]))
            Pr = row[mut_cols].astype(float).values
            Pr /= Pr.sum()

            # Multinomial
            muts_m = np.random.choice(96, N, replace=True, p=Pr)
            profile_m = np.bincount(muts_m, minlength=96) / N
            record_m = pd.Series(np.concatenate([profile_m, row[meta_cols].values]), 
                                 index=list(mut_cols) + list(meta_cols))
            record_m['boot_method'] = 'multinomial'
            record_m['boot'] = True
            XbootM.append(record_m)

            # Dirichlet
            alpha = (Pr + epsilon)
            alpha = alpha / alpha.sum() * dirichlet_strength
            Pr_dir = np.random.dirichlet(alpha)
            muts_d = np.random.choice(96, N, replace=True, p=Pr_dir)
            profile_d = np.bincount(muts_d, minlength=96) / N
            record_d = pd.Series(np.concatenate([profile_d, row[meta_cols].values]), 
                                 index=list(mut_cols) + list(meta_cols))
            record_d['boot_method'] = 'dirichlet'
            record_d['boot'] = True
            XbootD.append(record_d)

    # Append real samples
    data['boot_method'] = 'real'
    XbootM = pd.concat([data, pd.DataFrame(XbootM)], ignore_index=True)
    XbootD = pd.concat([data, pd.DataFrame(XbootD)], ignore_index=True)

    # Set index using Gene_KO_paired
    def make_index(df):
        return [f"{g}_{bm}_{suffix}" for g, bm in zip(df['Gene_KO_paired'], df['boot_method'])]
    
    XbootM.index = make_index(XbootM)
    XbootD.index = make_index(XbootD)

    return XbootM, XbootD


def parse_sample_metadata(index_series):
    """Extract gene, boot flag, and test/train status from sample names."""
    cleaned = index_series.str.replace(r'\.', '_', regex=True)
    parts = cleaned.str.split('_')

    gene_names = parts.str[0]
    boot_method = parts.str[1].str.lower()
    fold = parts.str[-1].str.lower()

    is_bootstrap = boot_method != 'real'
    is_test = fold == 'test'

    return gene_names, is_bootstrap, is_test


def generate_distinct_shades(base_cmap_name, n, min_val=0.3, max_val=0.8):
    cmap = cm.get_cmap(base_cmap_name)
    return [cmap(min_val + (max_val - min_val) * i / (n - 1)) for i in range(n)]






def make_umap_plot(X_real, X_boot, title, save_path=None, show=False):
    df = pd.concat([X_real, X_boot], axis=0)
    X_feats = df.iloc[:, :96].values

    gene_names   = df['Gene_KO']
    is_bootstrap = df['boot'].astype(bool)
    is_test      = (df['split'] == "test")

    reducer   = umap.UMAP(metric="cosine", random_state=42)
    embedding = reducer.fit_transform(X_feats)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Hardcoded gene colors
    gene_to_color = {
        "ATP2B4": "#1f77b4",  # Control (blue)
        "EXO1":   "#ad0101",  # HR (true red)
        "RNF168": "#ff2f14",  # HR (bright crimson)
        "UNG":    "#ffd500",  # BER (bright orange)
        "OGG1":   "#ff7b00",  # BER (deep orange)
        "MLH1":   "#33a833",  # MMR (bright green)
        "MSH2":   "#00ff55",  # MMR (lime green)
        "MSH6":   "#47f6d6",  # MMR (light mint)
        "PMS1":   "#8A8B8A",
        "PMS2":   "#4C4C4C",
    }

    for gene in gene_names.unique():
        color = gene_to_color.get(gene, 'gray')
        idx   = (gene_names == gene)

        for is_test_flag in (False, True):
            sub_idx  = idx & (is_test == is_test_flag)
            real_idx = sub_idx & (~is_bootstrap)
            boot_idx = sub_idx & (is_bootstrap)
            marker   = '^' if is_test_flag else 'o'

            # Bootstrapped (transparent, small)
            ax.scatter(
                embedding[boot_idx, 0], embedding[boot_idx, 1],
                c=color, edgecolors='black', linewidths=0.1, s=20, alpha=0.2, marker=marker
            )
            # Real (larger)
            ax.scatter(
                embedding[real_idx, 0], embedding[real_idx, 1],
                c=color, edgecolors='black', linewidths=0.8, s=30, alpha=0.95, marker=marker
            )

    # Legends
    gene_patches = [mpatches.Patch(color=gene_to_color.get(g, 'gray'), label=g)
                    for g in gene_names.unique()]
    gene_legend = ax.legend(
        handles=gene_patches, title="Gene (Color)",
        bbox_to_anchor=(1.01, 1.01), loc='upper left', fontsize='small'
    )
    ax.add_artist(gene_legend)

    shape_lines = [
        mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='Train', markersize=8),
        mlines.Line2D([], [], color='black', marker='^', linestyle='None', label='Test',  markersize=8)
    ]
    ax.legend(
        handles=shape_lines, title="Type (Shape)",
        bbox_to_anchor=(1.01, 0.3), loc='upper left', fontsize='small'
    )

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(False)
    ax.minorticks_on()
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig, ax
