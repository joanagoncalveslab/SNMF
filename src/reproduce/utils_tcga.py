# --- Standard Library ---
import os

# --- Scientific Computing ---
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import scipy.stats as stats

# --- Plotting ---
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, LinearSegmentedColormap, ListedColormap, Normalize
from adjustText import adjust_text
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, patches
from matplotlib.cm import ScalarMappable
from matplotlib.legend import Legend
import matplotlib.pyplot as plt
import matplotlib.patches as patches



# --- Stats and Multiple Testing ---
from statsmodels.stats import multitest
from statsmodels.stats.multitest import multipletests

# --- AnnData ---
from anndata import AnnData

def build_tcga_adata(cancer, project_root, bootstrap="dirichlet", lc_suffix=1):
    """
    Build TCGA AnnData object with associated annotations and predictions.

    Parameters
    ----------
    cancer : str
        Cancer type (e.g., 'BRCA').
    project_root : str
        Path to project root.
    bootstrap : str, optional
        Bootstrap type to use for LR predictions. Options: 'dirichlet' or 'multinomial'.
    lc_suffix : int, optional
        Lambda_c folder suffix (0 or 1). Default is 1.
    """
    # Map lc_suffix to actual lambda_c value
    lc_map = {0: 0.0, 1: 0.1}
    if lc_suffix not in lc_map:
        raise ValueError(f"lc_suffix must be 0 or 1, got {lc_suffix}")
    lambda_c_value = lc_map[lc_suffix]

    # --- Paths ---
    data_dir = os.path.join(project_root, 'data', 'processed', 'TCGA')

    # SNMF outputs (results from reproduce)
    snmf_dir = os.path.join(
        project_root, "results", "reproduce", "TCGA",
        f"{bootstrap}_lc{lc_suffix}", cancer, "SNMF_results"
    )

    # Model path (original bootstrap_comparison results)
    model_path = os.path.join(
        project_root, "results", "analysis", "bootstrap_comparison",
        f"SNMF_results_{bootstrap}_lc{lc_suffix}"
    )

    annot_dir = os.path.join(data_dir, 'annotations', cancer)

    # --- Load normalized profile ---
    profile_path = os.path.join(data_dir, 'profiles', 'norm_SBS', f"{cancer}.text")
    X = pd.read_csv(profile_path, sep='\t', index_col=0).T
    X.index = ['-'.join(s.split('-')[:3]) for s in X.index]

    # --- Load mutation counts ---
    count_path = os.path.join(data_dir, 'profiles', 'count', f"{cancer}.pkl")
    count_df = pd.read_pickle(count_path).iloc[:, :96]
    count_df.index = ['-'.join(s.split('-')[:3]) for s in count_df.index]
    count_df = count_df.loc[X.index]

    # --- Load pathway labels ---
    Y = pd.read_csv(os.path.join(annot_dir, 'Y_cancer.text'), sep='\t', index_col=0).T
    Y_hom = pd.read_csv(os.path.join(annot_dir, 'Y_cancer_hom.text'), sep='\t', index_col=0).T

    # --- Load gene matrices ---
    pathway_gene = pd.read_csv(os.path.join(data_dir, "pathway_gene_matrix.csv"), index_col=0)
    gene_list = pathway_gene.columns.tolist()

    gene_het = pd.read_csv(os.path.join(annot_dir, 'gene_matrix_het.text'), sep='\t', index_col=0)
    gene_hom = pd.read_csv(os.path.join(annot_dir, 'gene_matrix_hom.text'), sep='\t', index_col=0)

    gene_het = gene_het.reindex(columns=gene_list, index=X.index, fill_value=0)
    gene_hom = gene_hom.reindex(columns=gene_list, index=X.index, fill_value=0)

    # --- Build uns ---
    uns = {
        'mutation_count': count_df.sum(axis=1),
        'gene_het': gene_het,
        'gene_hom': gene_hom,
        'gene_colnames': gene_list,
        'pathway': pd.DataFrame(Y.loc[X.index].values, index=X.index, columns=Y.columns),
        'pathway_hom': pd.DataFrame(Y_hom.loc[X.index].values, index=X.index, columns=Y_hom.columns)
    }

    # SNMF predictions
    yhat_path = os.path.join(snmf_dir, 'Yhat_NMF.text')
    if os.path.exists(yhat_path):
        Yhat = pd.read_csv(yhat_path, sep='\t', index_col=0).T
        Yhat.index = ['-'.join(s.split('-')[:3]) for s in Yhat.index]
        Yhat = Yhat.loc[X.index]
        class_names = uns['pathway'].columns
        if len(class_names) == Yhat.shape[1]:
            Yhat.columns = class_names
        else:
            raise ValueError(f"Mismatch: Yhat has {Yhat.shape[1]} columns, but 'pathway' has {len(class_names)} labels.")
        uns['Yhat'] = Yhat

    # Exposures from SNMF
    etest_path = os.path.join(snmf_dir, "Exposures_test.txt")
    if os.path.exists(etest_path):
        exposures_test = pd.read_csv(etest_path, sep='\t', index_col=0)
        exposures_test.index = ['-'.join(s.split('-')[:3]) for s in exposures_test.index]
        exposures_test = exposures_test.loc[X.index]
        annotated_exposures_path = os.path.join(
            model_path, "run_0", "SBS96", "Suggested_Solution", "SBS96_De-Novo_Solution",
            "Activities", "Avg_Signature_Exposure_Annotated.txt"
        )
        if os.path.exists(annotated_exposures_path):
            annotated = pd.read_csv(annotated_exposures_path, sep="\t", index_col=0)
            if annotated.shape[1] == exposures_test.shape[1]:
                exposures_test.columns = annotated.columns
        uns['exposures_test'] = exposures_test

    # COSMIC exposures
    for cosmic_type in ["norm", "counts"]:
        c_dir = os.path.join(
            project_root, "results", "fit_cosmic", f"{cancer}_{cosmic_type}",
            "Assignment_Solution", "Activities"
        )
        c_file = os.path.join(c_dir, "Assignment_Solution_Activities.txt")
        if os.path.exists(c_file):
            exp = pd.read_csv(c_file, sep="\t", index_col=0)
            exp.index = ['-'.join(s.split('-')[:3]) for s in exp.index]
            uns[f'cosmic_exposures_{cosmic_type}'] = exp.loc[X.index]

    # Logistic regression predictions from COSMIC
    lr_pred_path = os.path.join(
        project_root, "results", "analysis", "bootstrap_comparison",
        f"LR_cosmic_{bootstrap.lower()}",
        f"{cancer}_predictions.csv"
    )
    if os.path.exists(lr_pred_path):
        prob_df = pd.read_csv(lr_pred_path, index_col=0)
        prob_df.index = ['-'.join(s.split('-')[:3]) for s in prob_df.index]
        uns['Yhat_LRcosmic'] = prob_df.loc[X.index].iloc[:, :4]

    # Optional: Ypred
    ypred_path = os.path.join(snmf_dir, f'Y_hat_N{X.shape[0]}.npy')
    if os.path.exists(ypred_path):
        Ypred = np.load(ypred_path)
        if Ypred.shape[1] == X.shape[0]:
            Ypred = Ypred.T
        uns['Ypred'] = pd.DataFrame(Ypred, index=X.index)

    # --- Create AnnData ---
    adata = AnnData(X.values, obs=pd.DataFrame(index=X.index), var=pd.DataFrame(index=X.columns))
    adata.obs_names = X.index
    adata.var_names = X.columns
    adata.uns = uns
    adata.layers["counts"] = count_df.values

    return adata


def build_tcga_multicancer_adata(cancer_list, project_root, bootstrap="dirichlet", lc_suffix=1):
    all_adata = []
    for cancer in cancer_list:
        print(f"→ Loading {cancer} ({bootstrap}_lc{lc_suffix})")
        adata = build_tcga_adata(cancer, project_root, bootstrap=bootstrap, lc_suffix=lc_suffix)
        adata.obs['cancer_type'] = cancer
        all_adata.append(adata)

    combined = all_adata[0].concatenate(
        all_adata[1:], 
        join='outer', 
        index_unique=None, 
        fill_value=0
    )

    combined.uns = {}
    all_keys = set().union(*[adata.uns.keys() for adata in all_adata])

    for key in all_keys:
        objs = [adata.uns[key] for adata in all_adata if key in adata.uns]
        if all(isinstance(obj, pd.DataFrame) for obj in objs):
            merged = pd.concat(objs, axis=0)
            merged = merged.loc[combined.obs_names.intersection(merged.index)]
            combined.uns[key] = merged
        elif all(isinstance(obj, pd.Series) for obj in objs):
            merged = pd.concat(objs, axis=0)
            merged = merged.loc[combined.obs_names.intersection(merged.index)]
            combined.uns[key] = merged
        else:
            combined.uns[key] = objs[0]

    return combined


def run_mannwhitney_gene_signature_test(
    adata,
    sig_matrix_key,          # e.g. "exposures_test"
    sig_col,                 # e.g. "SBS96A_HR"
    fdr_thresh=0.05,
    verbose=True,
    plot=False,
    project_root=None,       # NEW: to make save path absolute
    save_dir="results/figures/sup/tcga",  # relative to project_root
    ddr=None,                # e.g., "HR" or "MMR"
):
    # --- Checks ---
    if sig_matrix_key not in adata.uns:
        raise ValueError(f"{sig_matrix_key} not found in adata.uns.")
    if ddr is None:
        raise ValueError("Parameter 'ddr' must be provided (e.g., 'HR', 'MMR').")
    if "pathway" not in adata.uns:
        raise ValueError("Pathway labels not found in adata.uns['pathway'].")

    # --- Signature values ---
    matrix_df = pd.DataFrame(adata.uns[sig_matrix_key], index=adata.obs_names)
    if sig_col not in matrix_df.columns:
        raise ValueError(f"{sig_col} not found in adata.uns['{sig_matrix_key}'] columns.")
    sig_values = matrix_df[sig_col].values

    # --- Pathway labels (to define WT baseline outside the DDR class) ---
    pathway_df = pd.DataFrame(adata.uns["pathway"], index=adata.obs_names)
    if ddr not in pathway_df.columns:
        raise ValueError(f"DDR class '{ddr}' not found in pathway columns.")
    pathway_labels = pathway_df[ddr]

    # --- Mutation matrices ---
    gene_het = pd.DataFrame(adata.uns['gene_het'], index=adata.obs_names, columns=adata.uns['gene_colnames'])
    gene_hom = pd.DataFrame(adata.uns['gene_hom'], index=adata.obs_names, columns=adata.uns['gene_colnames'])
    all_genes = sorted(set(gene_het.columns).union(set(gene_hom.columns)))
    gene_het = gene_het.reindex(columns=all_genes, fill_value=0)
    gene_hom = gene_hom.reindex(columns=all_genes, fill_value=0)
    gene_mut = (gene_het > 0) | (gene_hom > 0)

    # --- Mann–Whitney tests ---
    results = []
    for gene in all_genes:
        mut_status = gene_mut[gene].values.astype(bool)
        if mut_status.sum() == 0:
            continue

        # WT = wild-type AND outside the DDR class (pathway_labels == 0)
        wt_mask = (~mut_status) & (pathway_labels == 0)
        mut_scores = sig_values[mut_status]
        wt_scores  = sig_values[wt_mask]

        if len(wt_scores) == 0 or len(mut_scores) == 0:
            continue

        stat_greater, pval_greater     = mannwhitneyu(mut_scores, wt_scores, alternative="greater")
        stat_less,    pval_less        = mannwhitneyu(wt_scores,  mut_scores, alternative="greater")
        stat_twosided, pval_twosided   = mannwhitneyu(mut_scores, wt_scores, alternative="two-sided")

        results.append({
            "Gene": gene,
            "MW_stat_twosided":   stat_twosided,
            "MW_pval_twosided":   pval_twosided,
            "MW_stat_mut_gt_wt":  stat_greater,
            "MW_pval_mut_gt_wt":  pval_greater,
            "MW_stat_wt_gt_mut":  stat_less,
            "MW_pval_wt_gt_mut":  pval_less,
            "N_mut": int(mut_status.sum()),
            "N_wt":  int(wt_mask.sum()),
        })

    if not results:
        print(f"No valid genes to test for {sig_col}.")
        return adata

    results_df = pd.DataFrame(results).set_index("Gene")

    # --- FDR corrections ---
    results_df["MW_fdr_twosided"]   = multitest.fdrcorrection(results_df["MW_pval_twosided"])[1]
    results_df["MW_fdr_mut_gt_wt"]  = multitest.fdrcorrection(results_df["MW_pval_mut_gt_wt"])[1]
    results_df["MW_fdr_wt_gt_mut"]  = multitest.fdrcorrection(results_df["MW_pval_wt_gt_mut"])[1]

    # Store in AnnData
    adata.uns[f"MW_results_{sig_col}"] = results_df

    if verbose:
        sig_twosided = results_df.query("MW_fdr_twosided < @fdr_thresh").index.tolist()
        sig_mut_gt_wt = results_df.query("MW_fdr_mut_gt_wt < @fdr_thresh").index.tolist()
        sig_wt_gt_mut = results_df.query("MW_fdr_wt_gt_mut < @fdr_thresh").index.tolist()
        print(f"Significant genes for {sig_col}:")
        print("  Two-sided:", sig_twosided)
        print("  Mut > WT :", sig_mut_gt_wt)
        print("  WT > Mut :", sig_wt_gt_mut)

    saved_paths = []
    if plot:
        # Resolve save root
        if project_root is None:
            project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
        ddr_folder = ddr if ddr is not None else "unknown_ddr"
        plot_dir = os.path.join(project_root, save_dir, ddr_folder)
        os.makedirs(plot_dir, exist_ok=True)

        # Genes with MUT>WT significance
        sig_genes = (
            results_df.loc[results_df["MW_fdr_mut_gt_wt"] < fdr_thresh]
            .sort_values("MW_fdr_mut_gt_wt")
            .index.tolist()
        )

        if len(sig_genes) == 0:
            print("No significantly upregulated genes to plot.")
        else:
            sns.set(style="whitegrid")
            n_genes = len(sig_genes)
            fig = plt.figure(figsize=(2.2 * n_genes, 6))

            for i, gene in enumerate(sig_genes, start=1):
                mut_status = gene_mut[gene].values.astype(bool)
                wt_mask    = (~mut_status) & (pathway_labels == 0)

                plot_df = pd.DataFrame({
                    "Signature_Value": np.concatenate([sig_values[wt_mask], sig_values[mut_status]]),
                    "Mutation_Status": (["WT"] * wt_mask.sum()) + (["MUT"] * mut_status.sum()),
                })

                ax = plt.subplot(1, n_genes, i)
                order = ["WT", "MUT"]
                sns.boxplot(
                    x="Mutation_Status", y="Signature_Value", data=plot_df,
                    order=order, palette={"WT": "green", "MUT": "orangered"}, ax=ax
                )
                sns.stripplot(
                    x="Mutation_Status", y="Signature_Value", data=plot_df,
                    order=order, color="black", alpha=0.3, jitter=True, ax=ax
                )

                # Significance stars (FDR)
                pval = results_df.loc[gene, "MW_fdr_mut_gt_wt"]
                stars = "***" if pval <= 1e-3 else "**" if pval <= 1e-2 else "*" if pval <= 5e-2 else ""
                y_max, y_min = plot_df["Signature_Value"].max(), plot_df["Signature_Value"].min()
                y, h = y_max + 0.05 * (y_max - y_min), 0.05 * (y_max - y_min)
                ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], lw=1.5, c="k")
                ax.text(0.5, y + 0.5 * h, stars, ha="center", va="bottom", fontsize=18, color="k")

                # Labels
                mut_count = int(mut_status.sum())
                ax.set_title("")
                ax.set_xlabel(gene)
                ax.set_xticklabels(["WT", f"MUT ({mut_count})"])
                if i > 1:
                    ax.set_ylabel("")
                else:
                    ax.set_ylabel(sig_col)

            fig.tight_layout()

            # Build filenames that won’t collide
            adata_type = adata.uns.get("adata_type", "adata")
            base = f"{adata_type}_{ddr}_{sig_col}_MW_boxplots"
            pdf_path = os.path.join(plot_dir, f"{base}.pdf")
            svg_path = os.path.join(plot_dir, f"{base}.svg")

            fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
            # fig.savefig(svg_path, format="svg", bbox_inches="tight")
            plt.show()
            plt.close(fig)

            print(f"Saved boxplots to:\n  {pdf_path}\n  {svg_path}")
            saved_paths.extend([pdf_path, svg_path])

    # Optionally store where we saved things
    adata.uns[f"MW_results_{sig_col}_paths"] = saved_paths
    return adata


def plot_ddr_signature(
    adata,
    genes,
    ddr_label,
    project_root=None,
    save_dir="results/figures/sup/tcga",
    save_basename=None,   # optional file name stem; if None we auto-name
    # --- keep your original defaults/behavior ---
):
    def as_df(obj, index, columns=None):
        if isinstance(obj, pd.DataFrame):
            return obj
        else:
            return pd.DataFrame(obj, index=index, columns=columns)

    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))

    # -------- your original code (unchanged) ----------
    samples = adata.obs_names
    gene_het = as_df(adata.uns['gene_het'], index=samples, columns=adata.uns.get('gene_colnames'))
    gene_hom = as_df(adata.uns['gene_hom'], index=samples, columns=adata.uns.get('gene_colnames'))

    # --- Build status matrix ---
    status_matrix_df = pd.DataFrame('WT', index=samples, columns=genes)
    for gene in genes:
        if gene in gene_hom.columns:
            status_matrix_df.loc[gene_hom[gene] > 0, gene] = 'HOM'
        if gene in gene_het.columns:
            is_het = gene_het[gene] > 0
            not_hom = status_matrix_df[gene] != 'HOM'
            status_matrix_df.loc[is_het & not_hom, gene] = 'HET'
    status_map = {'WT': 0, 'HET': 1, 'HOM': 2}
    status_matrix = status_matrix_df.replace(status_map).T.values  # genes x samples

    # --- Pathway exposures ---
    exposures_df = as_df(
        adata.uns['exposures_test'],
        index=samples,
        columns=adata.uns.get('exposures_test').columns if isinstance(adata.uns.get('exposures_test'), pd.DataFrame) else None
    )
    matching_cols = [c for c in exposures_df.columns if len(c.split('_')) > 1 and c.split('_')[1] == ddr_label]
    if not matching_cols:
        raise ValueError(f"No signature found in exposures_test for ddr_label='{ddr_label}'")
    sig_col = matching_cols[0]
    E_pathway = exposures_df.loc[samples, sig_col].values

    Yhat_df = as_df(
        adata.uns['Yhat'],
        index=samples,
        columns=adata.uns.get('Yhat').columns if isinstance(adata.uns.get('Yhat'), pd.DataFrame) else None
    )
    if ddr_label not in Yhat_df.columns:
        raise ValueError(f"{ddr_label} not found in Yhat")
    Y_pathway = Yhat_df.loc[samples, ddr_label].values

    # --- Sort samples ---
    order = np.argsort(E_pathway)
    E_pathway = E_pathway[order]
    Y_pathway = Y_pathway[order]
    status_matrix = status_matrix[:, order]

    # --- COSMIC and mutation count ---
    cosmic_sig_map = {
        "HR": ["SBS3", "SBS8"],
        "MMR": ["SBS6", "SBS14", "SBS15", "SBS20", "SBS21", "SBS26", "SBS44"]
    }
    selected_sigs = cosmic_sig_map.get(ddr_label, [])
    cosmic_df = as_df(
        adata.uns['cosmic_exposures_norm'],
        index=samples,
        columns=adata.uns.get('cosmic_exposures_norm').columns if isinstance(adata.uns.get('cosmic_exposures_norm'), pd.DataFrame) else None
    )
    selected_sigs_in_data = [s for s in selected_sigs if s in cosmic_df.columns]
    if selected_sigs_in_data:
        cosmic_data = cosmic_df.loc[samples[order], selected_sigs_in_data].values.T
        cosmic_data = cosmic_data / (cosmic_data.max(axis=1, keepdims=True) + 1e-8)
    else:
        cosmic_data = np.zeros((0, len(samples)))  # empty

    mut_count = np.array(adata.uns['mutation_count']).reshape(-1)[order].astype(float)
    #! Few mutation count outlier skew range...
    mut_count_max = np.percentile(mut_count, 99.8)
    if mut_count_max == 0:
        mut_count_max = 1
    mut_count_norm = mut_count
    mut_count_row = mut_count_norm.reshape(1, -1)

    # --- Labels ---
    row_labels_cat = genes
    row_labels_cont = selected_sigs_in_data + ["Mutation count"]

    # --- Colormaps ---
    cmap_cat = ['ghostwhite', 'darkorange', 'darkred']  # WT, HET, HOM
    light_to_purple = LinearSegmentedColormap.from_list("gray_to_purple", ["#f8f8f8", "#6a0dad"])
    cmap_blue = LinearSegmentedColormap.from_list("gray_to_blue", ["#f8f8f8", "#1f77b4"])

    # --- Figure ---
    n_cosmic = len(selected_sigs_in_data)
    heights = [2, max(1, len(genes)), max(1, n_cosmic), 0.6]
    fig, axes = plt.subplots(4, 1, gridspec_kw={'height_ratios': heights}, figsize=(6.2, 6), sharex=True)

    # 1) E/Y heatmap
    top = np.vstack([E_pathway, Y_pathway])
    sns.heatmap(top, ax=axes[0], cmap='viridis', cbar=False, vmin=0, vmax=1)
    axes[0].set_yticks([0.5, 1.5])
    axes[0].set_yticklabels([f"$E_{{{ddr_label}}}$", f"$Y_{{{ddr_label}}}$"], rotation=0)
    axes[0].hlines([1], *axes[0].get_xlim(), color='white')
    axes[0].set_xticks([])
    axes[0].tick_params(left=False, bottom=False)

    # 2) Mutation status heatmap
    sns.heatmap(status_matrix, ax=axes[1], cmap=cmap_cat, cbar=False, linewidth=0)
    axes[1].set_yticks(np.arange(len(row_labels_cat)) + 0.5)
    axes[1].set_yticklabels(row_labels_cat, rotation=0)
    axes[1].set_xticks([])
    axes[1].tick_params(left=False, bottom=False)

    # 3) COSMIC exposures heatmap
    if n_cosmic > 0:
        sns.heatmap(cosmic_data, ax=axes[2], cmap=light_to_purple, cbar=False, vmin=0, vmax=1)
        axes[2].set_yticks(np.arange(n_cosmic) + 0.5)
        axes[2].set_yticklabels(selected_sigs_in_data, rotation=0)
    else:
        axes[2].axis('off')
    axes[2].set_xticks([])
    axes[2].tick_params(left=False, bottom=False)

    # 4) Mutation count heatmap
    sns.heatmap(mut_count_row, ax=axes[3], cmap=cmap_blue, cbar=False, vmin=0, vmax=mut_count_max)
    axes[3].set_yticks([0.5])
    axes[3].set_yticklabels(["Mutation count"], rotation=0)
    axes[3].set_xticks([])
    axes[3].tick_params(left=False, bottom=False)
    axes[3].set_xlabel(f'{status_matrix.shape[1]} samples ({ddr_label})')

    plt.subplots_adjust(hspace=0.05, left=0.06, right=0.78)

    # Legend column
    legend_left_cat = 0.81
    legend_left_cbar = 0.89
    legend_bottom = -0.18
    legend_top = 0.86
    total_h = legend_top - legend_bottom
    legend_pieces = [0.15, 0.04, 0.15, 0.04, 0.15, 0.04, 0.15]
    legend_width_cat = 0.15
    legend_width_cbar = 0.03

    y0 = legend_top
    cbar_axes = []
    for i, frac in enumerate(legend_pieces):
        h = frac * total_h
        y0 -= h
        if i in [1, 3, 5]:
            cbar_axes.append(None)
            continue
        elif i == 2:
            ax_legend_piece = fig.add_axes([legend_left_cat, y0, legend_width_cat, h])
        else:
            ax_legend_piece = fig.add_axes([legend_left_cbar, y0, legend_width_cbar, h])
        ax_legend_piece.clear()
        cbar_axes.append(ax_legend_piece)

    # 1) viridis colorbar (top)
    cax0 = cbar_axes[0]
    cax0.clear()
    cbar0 = fig.colorbar(cm.ScalarMappable(norm=Normalize(0, 1), cmap='viridis'), cax=cax0, orientation='vertical')
    cbar0.ax.yaxis.set_ticks_position('left')
    cbar0.set_label('E / Y (0–1)', labelpad=3, fontsize=9)
    cbar0.ax.tick_params(labelsize=8)
    cax0.yaxis.set_ticks_position('left')
    cax0.axis('on')

    # 2) categorical legend (mutation status)
    cax_cat = cbar_axes[2]
    cax_cat.clear()
    cax_cat.axis('on')
    cax_cat.set_xticks([])
    cax_cat.set_yticks([])
    legend_patches = [patches.Patch(color=['ghostwhite', 'darkorange', 'darkred'][i], label=l)
                      for i, l in enumerate(['WT', 'HET', 'HOM'])]
    cax_cat.legend(handles=legend_patches, loc='center', frameon=False, title='Mutation status', fontsize=9, title_fontsize=10)

    # 3) COSMIC exposures colorbar
    cax_cosmic = cbar_axes[4]
    if cax_cosmic is not None:
        cax_cosmic.clear()
        if len(selected_sigs_in_data) > 0:
            cbar2 = fig.colorbar(cm.ScalarMappable(norm=Normalize(0, 1), cmap=light_to_purple), cax=cax_cosmic, orientation='vertical')
            cbar2.ax.yaxis.set_ticks_position('left')
            cbar2.set_label('COSMIC exposure', labelpad=3, fontsize=9)
            cbar2.ax.tick_params(labelsize=8)
            cax_cosmic.yaxis.set_ticks_position('left')
            cax_cosmic.axis('on')
        else:
            cax_cosmic.axis('off')

    # 4) Mutation count colorbar
    cax_mut = cbar_axes[6]
    cax_mut.clear()
    cbar3 = fig.colorbar(cm.ScalarMappable(norm=Normalize(0, mut_count_max), cmap=cmap_blue), cax=cax_mut, orientation='vertical')
    cbar3.ax.yaxis.set_ticks_position('left')
    cbar3.set_label('Mutation count', labelpad=3, fontsize=9)
    cbar3.ax.tick_params(labelsize=8)
    cax_mut.yaxis.set_ticks_position('left')
    cax_mut.axis('on')

    # -------- saving (PDF only) ----------
    if save_basename is None:
        # Example: HR_EYcosmic_mutMatrix_5genes
        save_basename = f"{ddr_label}_EYcosmic_mutMatrix_{len(genes)}genes"

    out_dir = os.path.join(project_root, save_dir, ddr_label)
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{save_basename}.pdf")

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved figure to {pdf_path}")

    plt.show()
    plt.close(fig)

    return pdf_path


def get_significant_genes_from_adata_df(adata, mw_results_key, fdr_thresh=0.05):
    """
    Extract significant genes from MW test results stored in adata.uns[mw_results_key],
    sorted by increasing 'MW_fdr_mut_gt_wt'.

    Parameters:
        adata: AnnData object
        mw_results_key: str
            Key in adata.uns where the MW test result DataFrame is stored.
        fdr_thresh: float
            Significance threshold on 'MW_fdr_mut_gt_wt'.

    Returns:
        significant_genes: list of gene names passing the FDR threshold, sorted by significance
    """
    df = adata.uns.get(mw_results_key)
    if df is None:
        raise KeyError(f"Key '{mw_results_key}' not found in adata.uns")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"adata.uns['{mw_results_key}'] is not a pandas DataFrame")
    if 'MW_fdr_mut_gt_wt' not in df.columns:
        raise ValueError(f"Expected column 'MW_fdr_mut_gt_wt' not found in DataFrame at '{mw_results_key}'")

    # Filter and sort by significance
    signif_df = df[df['MW_fdr_mut_gt_wt'] <= fdr_thresh].sort_values(by='MW_fdr_mut_gt_wt')
    return signif_df.index.to_list()



def plot_mw_gene_stats(
    adata,
    mw_sig="SBS96A_HR",
    pval_threshold=0.01,
    annotate=True,
    figsize=(4, 4),
    project_root=None,              
    save_dir="results/figures/sup/tcga",  # relative to project_root
    verbose=False
):
    key = f'MW_results_{mw_sig}'
    if key not in adata.uns:
        raise ValueError(f"'{key}' not found in adata.uns")

    df = adata.uns[key].copy()
    required_cols = ['MW_stat_mut_gt_wt', 'MW_stat_wt_gt_mut', 'MW_fdr_mut_gt_wt', 'N_mut']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing one of required columns: {required_cols}")

    x, y, hue, size = 'MW_stat_wt_gt_mut', 'MW_stat_mut_gt_wt', 'MW_fdr_mut_gt_wt', 'N_mut'

    # figure save path (PDF only), infer DDR from mw_sig suffix
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    ddr_label = mw_sig.split("_")[-1] if "_" in mw_sig else "DDR"
    out_dir = os.path.join(project_root, save_dir, ddr_label)
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"MW_gene_stats_{mw_sig}.pdf")

    style_rc = {
        'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.linewidth': 1.0, 'grid.linewidth': 0., 'lines.linewidth': 1.5,
        'lines.markersize': 6.0, 'patch.linewidth': 1.0,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
        'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
        'xtick.minor.size': 2.0, 'ytick.minor.size': 2.0
    }

    annotated = []
    with sns.axes_style('whitegrid', rc={
        'axes.edgecolor': 'black', 'axes.facecolor': 'white',
        'xtick.bottom': True, 'ytick.left': True,
        'xtick.direction': 'in', 'ytick.direction': 'in',
    }), sns.plotting_context(font_scale=1, rc=style_rc):

        # main figure with room for legends
        fig = plt.figure(figsize=(figsize[0]+1.8, figsize[1]))
        ax = fig.add_axes([0.1, 0.15, 0.65, 0.75])

        # size legend basis (number of mutated tumors)
        scatter_sizes = [10, 50, 100, 200, 400]
        max_marker_area = 150
        size_scale = max_marker_area / max(scatter_sizes)
        scaled_sizes = [s * size_scale for s in scatter_sizes]

        sc = sns.scatterplot(
            data=df, x=x, y=y, hue=hue, size=size,
            sizes=(scaled_sizes[0], scaled_sizes[-1]),
            palette='flare_r', edgecolor='white',
            linewidth=0.2, size_norm=(0, max(scatter_sizes)),
            hue_norm=LogNorm(vmin=1e-4, vmax=1),
            legend=False, ax=ax, alpha=0.9
        )

        # annotate significant genes
        texts = []
        if annotate:
            for idx, row in df.iterrows():
                if row[hue] < pval_threshold:
                    texts.append(ax.text(row[x], row[y], idx, ha='center', size=5, color='black', weight='light'))
                    annotated.append((idx, row[y]))
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='black', lw=0.3))
            if verbose:
                print(f"Genes with FDR < {pval_threshold}:")
                for g, val in annotated:
                    print(f"  {g}: MW_stat_mut_gt_wt = {val:.2f}")

        ax.axline((0, 0), slope=1, color="b", dashes=(5, 2))
        ax.set(
            xlabel=f'MW statistic WT > MUT ($E_{{{mw_sig}}}$)',
            ylabel=f'MW statistic MUT > WT ($E_{{{mw_sig}}}$)'
        )
        ax.grid(False)

        # legend/colorbar column
        legend_left_cbar = 0.87
        legend_bottom = 0.12
        legend_top = 0.85
        total_h = legend_top - legend_bottom
        legend_pieces = [0.45, 0.15, 0.45]

        y0 = legend_top
        cbar_axes = []
        for frac in legend_pieces:
            h = frac * total_h
            y0 -= h
            cax = fig.add_axes([legend_left_cbar, y0, 0.04, h])
            cax.clear()
            cbar_axes.append(cax)

        # 1) p-value colorbar
        cax0 = cbar_axes[0]
        cbar0 = fig.colorbar(cm.ScalarMappable(norm=LogNorm(vmin=1e-4, vmax=1), cmap='flare_r'),
                             cax=cax0, orientation='vertical')
        cbar0.ax.yaxis.set_ticks_position('left')
        cbar0.set_label('p-value MW test', labelpad=3, fontsize=9)
        cbar0.ax.tick_params(labelsize=8)

        # 2) spacer
        cbar_axes[1].axis('off')

        # 3) size legend (mutated tumors)
        cax2 = cbar_axes[2]
        cax2.clear()
        cax2.axis('off')
        base_y, spacing = 0.1, 0.14
        y_positions = [base_y + i * spacing for i in range(len(scaled_sizes))]
        circle_x, label_x = 0.40, 0.95
        for y, sz in zip(y_positions, scaled_sizes):
            cax2.scatter(circle_x, y, s=sz, color='gray', alpha=0.6, edgecolor='black')
        for y, label in zip(y_positions, scatter_sizes):
            cax2.text(label_x, y, str(label), va='center', fontsize=8)
        cax2.set_xlim(0, 1); cax2.set_ylim(0, 1)
        cax2.text(0.5, base_y + spacing * len(scaled_sizes) + 0.05,
                  'Number of tumors with \n respective gene mutated',
                  ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # --- SAVE (PDF only) ---
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Saved figure to {pdf_path}")

        plt.show()
        plt.close(fig)

    return annotated
