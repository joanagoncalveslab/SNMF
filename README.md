<h1><br>
<a href="https://github.com/joanagoncalveslab/SNMF/">SNMF</a></h1>

<h4 align="center">

SNMF: Integrated learning of mutational signatures and prediction of DNA
repair deficiencies

</h4>

<p><a href="https://github.com/joanagoncalveslab/SNMF/-/commits/master">
<img src="https://img.shields.io/badge/last%20commit-februari-yellow" alt="GitHub last commit"/></a>
</p>

<p><a href="#abstract">Abstract</a> •
<a href="#repository-description">Repository Description</a> •
<a href="#license">License</a></p>

------------------------------------------------------------------------

## Abstract
Motivation: Many tumours show deficiencies in DNA damage response (DDR), which influences tumorigenesis and progression, but also exposes vulnerabilities with therapeutic potential. Assessing which patients might benefit from DDR-targeting therapy requires knowledge of tumour DDR deficiency status, with mutational signatures reportedly better predictors than loss of function mutations in select genes. However, signatures are identified independently using unsupervised learning, which is not optimised to distinguish between different pathway or gene deficiencies. Results: We propose SNMF, a supervised non- negative matrix factorisation that jointly optimises the learning of signatures: (1) shared across samples, and (2) predictive of DDR deficiency. We applied SNMF to mutation profiles of human induced pluripotent cell lines carrying gene knockouts linked to three DDR pathways. The SNMF model achieved high accuracy (0.971) and learned more complete signatures of the DDR status of a sample, further discerning distinct mechanisms within a pathway. Cell line SNMF signatures recapitulated tumour derived COSMIC signatures and predicted DDR pathway deficiency of TCGA tumours with high recall, suggesting that SNMF-like models can leverage libraries of induced DDR deficiencies to decipher intricate DDR signatures underlying patient tumours. Code: <https://github.com/joanagoncalveslab/SNMF>.                    

## SNMF model

![](fig1_SNMF.pdf)

## Repository Description

##### Folder hierarchy:

-   **data**: Includes all the data for feature generation or
    experiments.
    -   **raw**: raw repair deficient cell line data (zou2021)
    -   **processed**: bootstrapped cell line data and TCGA mutational
        profiles
-   **results**:
    -   **EDA**: exploratory data analysis

    -   **final**: result from paper
-   **SNMF**: containing all the code for the SNMF model, adapted from
    the SigProfiler framework
    -   **src**: containing test.py to run the SNMF method
-   **src**:
    -   **processing**: code for preprocessing (bootstrapping) of data

## License

[![License: BSD
2-Clause](https://img.shields.io/badge/License-DSB%202%20Clause-blue.svg?style=flat-square)](https://tldrlegal.com/license/gnu-lesser-general-public-license-v3-(lgpl-3))

-   Copyright © [Sander-Goossens].
