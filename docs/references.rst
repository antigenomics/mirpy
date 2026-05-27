References
==========

The algorithms and data resources implemented in mirpy are described in the
following publications. Please cite the relevant papers when using mirpy in
your work.

Core Methods
------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Method
     - Citation
   * - **ALICE** — Poisson neighbourhood enrichment for antigen-driven TCR clusters
     - Pogorelyy MV, Minervina AA, Shugay M *et al.* (2019)
       Detecting T cell receptors involved in immune responses from single
       repertoire snapshots.
       *PLoS Biol.* 17(6):e3000314.
       `PMID:31194732 <https://pubmed.ncbi.nlm.nih.gov/31194732/>`_
   * - **TCRNET** — MC-control neighbourhood enrichment
     - Lupyr KR *et al.* (2025)
       Neighborhood enrichment for the identification of antigen-specific
       T-cell receptors.
       *Brief. Bioinform.* 26(5):bbaf495.
       `PMID:40996146 <https://pubmed.ncbi.nlm.nih.gov/40996146/>`_
   * - **TCREmp** — prototype-based TCR embeddings
     - Kremlyakova Y *et al.* (2025)
       TCREMP: A Bioinformatic Pipeline for Efficient Embedding of T-cell
       Receptor Sequences from Immune Repertoire and Single-cell Sequencing Data.
       *J. Mol. Biol.* 437(15):169205.
       `PMID:40368275 <https://pubmed.ncbi.nlm.nih.gov/40368275/>`_
   * - **TCRdist** — weighted V-gene + CDR3 alignment distance
     - Dash P *et al.* (2017)
       Quantifiable predictive features define epitope-specific T cell receptor
       repertoires.
       *Nature* 547(7661):89–93.
       `PMID:28636592 <https://pubmed.ncbi.nlm.nih.gov/28636592/>`_
   * - **GLIPH** — CDR3 motif clustering for antigen specificity groups
     - Glanville J *et al.* (2017)
       Identifying specificity groups in the T cell receptor repertoire.
       *Nature* 547(7661):94–98.
       `PMID:28636589 <https://pubmed.ncbi.nlm.nih.gov/28636589/>`_
   * - **GLIPH2** — large-scale CDR3 clustering (millions of TCRs)
     - Huang H *et al.* (2020)
       Analyzing the Mycobacterium tuberculosis immune response by T-cell
       receptor clustering with GLIPH2 and genome-wide antigen screening.
       *Nat. Biotechnol.* 38(10):1194–1202.
       `PMID:32341563 <https://pubmed.ncbi.nlm.nih.gov/32341563/>`_
   * - **Diversity indices** (Shannon, Chao1, Gini-Simpson, rarefaction)
     - Shugay M *et al.* (2015)
       VDJtools: Unifying Post-analysis of T Cell Receptor Repertoires.
       *PLoS Comput. Biol.* 11(11):e1004503.
       `PMID:26606115 <https://pubmed.ncbi.nlm.nih.gov/26606115/>`_
   * - **CDR3 motif logos** — IC and selection logos vs OLGA background
     - Pogorelyy MV, Minervina AA, Shugay M *et al.* (2019) (same as ALICE paper above)
       `PMID:31194732 <https://pubmed.ncbi.nlm.nih.gov/31194732/>`_

Databases and Annotation
------------------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Resource
     - Citation
   * - **VDJdb** — curated antigen-specific TCR database (original)
     - Shugay M *et al.* (2018)
       VDJdb: a curated database of T-cell receptor sequences with known
       antigen specificity.
       *Nucleic Acids Res.* 46(D1):D419–D427.
       `PMID:28977646 <https://pubmed.ncbi.nlm.nih.gov/28977646/>`_
   * - **VDJdb 2019** — database extension and T-cell receptor motif compendium
     - Bagaev DV *et al.* (2020)
       VDJdb in 2019: database extension, new analysis infrastructure and a
       T-cell receptor motif compendium.
       *Nucleic Acids Res.* 48(D1):D1057–D1062.
       `PMID:31588507 <https://pubmed.ncbi.nlm.nih.gov/31588507/>`_
   * - **VDJdb** — SARS-CoV-2 expansion
     - Goncharov M *et al.* (2022)
       VDJdb in the pandemic era: a compendium of T cell receptors specific
       for SARS-CoV-2.
       *Nat. Methods* 19(9):1017–1019.
       `PMID:35970936 <https://pubmed.ncbi.nlm.nih.gov/35970936/>`_
   * - **Antigen-specificity annotation** framework for high-throughput studies
     - Pogorelyy MV, Shugay M (2019)
       A Framework for Annotation of Antigen Specificities in
       High-Throughput T-Cell Repertoire Sequencing Studies.
       *Front. Immunol.* 10:2159.
       `PMID:31616409 <https://pubmed.ncbi.nlm.nih.gov/31616409/>`_

Biological Contexts
-------------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Context
     - Citation
   * - **T cell repertoire aging** — 79-donor aging cohort (VDJtools format, used in notebooks)
     - Britanova OV *et al.* (2016)
       Dynamics of Individual T Cell Repertoires: From Cord Blood to Centenarians.
       *J. Immunol.* 196(12):5005–5013.
       `PMID:27183615 <https://pubmed.ncbi.nlm.nih.gov/27183615/>`_
   * - **Pre-immune antigen-specific landscape** — naive TCR Pgen context
     - Pogorelyy MV *et al.* (2018)
       Exploring the pre-immune landscape of antigen-specific T cells.
       *Genome Med.* 10:68.
       `PMID:30144804 <https://pubmed.ncbi.nlm.nih.gov/30144804/>`_
   * - **Regulatory T cell repertoire** expansion mechanism
     - Feng Y *et al.* (2015)
       A mechanism for expansion of regulatory T-cell repertoire and its role
       in self-tolerance.
       *Nature* 528(7580):132–136.
       `PMID:26605529 <https://pubmed.ncbi.nlm.nih.gov/26605529/>`_
   * - **TCR structural modeling** — template-based large-scale approach
     - Shcherbinin DS *et al.* (2023)
       Large-scale template-based structural modeling of T-cell receptors with
       known antigen specificity reveals complementarity features.
       *Front. Immunol.* 14:1224969.
       `PMID:37649481 <https://pubmed.ncbi.nlm.nih.gov/37649481/>`_
   * - **TCRen** — structure-based prediction of TCR–epitope recognition
     - Karnaukhov VK *et al.* (2024)
       Structure-based prediction of T cell receptor recognition of unseen
       epitopes using TCRen.
       *Nat. Comput. Sci.* 4(7):510–521.
       `PMID:38987378 <https://pubmed.ncbi.nlm.nih.gov/38987378/>`_
   * - **Thymic selection** — αβ TCR repertoire signatures
     - Luppov DV *et al.* (2025)
       Comprehensive analysis of αβT-cell receptor repertoires reveals
       signatures of thymic selection.
       *Front. Immunol.* 16:1605170.
       `PMID:41050667 <https://pubmed.ncbi.nlm.nih.gov/41050667/>`_
   * - **Clonal T cell tracking** in cancer neoantigen responses
     - Shagina IA *et al.* (2026)
       Capturing and tracking clonal T cell response to cancer neoantigens.
       *Cancer Immunol. Res.*
       `PMID:41843768 <https://pubmed.ncbi.nlm.nih.gov/41843768/>`_
   * - **Immunogenomics diversity** — ancestral representation
     - Peng K *et al.* (2021)
       Diversity in immunogenomics: the value and the challenge.
       *Nat. Methods* 18(6):588–591.
       `PMID:34002093 <https://pubmed.ncbi.nlm.nih.gov/34002093/>`_
   * - **High-resolution repertoire analysis** — bystander Tfh/Tfr activation
     - Ritvo P-G *et al.* (2018)
       High-resolution repertoire analysis reveals a major bystander activation
       of Tfh and Tfr cells.
       *Proc. Natl. Acad. Sci. USA* 115(38):9604–9609.
       `PMID:30158170 <https://pubmed.ncbi.nlm.nih.gov/30158170/>`_
   * - **T cell clonal expansions** — single-cell deep profiling
     - Pavlova AV *et al.* (2024)
       Detecting T-cell clonal expansions and quantifying clone survival using
       deep profiling of immune repertoires.
       *Front. Immunol.* 15:1321603.
       `PMID:38633256 <https://pubmed.ncbi.nlm.nih.gov/38633256/>`_

Complete Publication List
-------------------------

All publications co-authored by Mikhail Shugay that are relevant to mirpy
functionality or provide biological context for AIRR-seq analyses.
Titles are exactly as in PubMed. Excludes MiXCR (PMID:25924071) and MiTCR
(PMID:23892897).

.. list-table::
   :widths: 30 40 30
   :header-rows: 1

   * - Topic (mirpy relevance)
     - Title
     - Citation / PMID
   * - SARS-CoV-2 T cell biomarkers via repertoire profiling
     - Inference of SARS-CoV-2 exposure biomarkers using large-scale T-cell
       repertoire profiling.
     - Vlasova AV *et al.* *Genome Med.* 2026.
       `PMID:41680899 <https://pubmed.ncbi.nlm.nih.gov/41680899/>`_
   * - Clonal T cell tracking in cancer neoantigen responses
     - Capturing and tracking clonal T cell response to cancer neoantigens.
     - Shagina IA *et al.* *Cancer Immunol. Res.* 2026.
       `PMID:41843768 <https://pubmed.ncbi.nlm.nih.gov/41843768/>`_
   * - Thymic selection signatures in αβ TCR repertoires
     - Comprehensive analysis of αβT-cell receptor repertoires reveals
       signatures of thymic selection.
     - Luppov DV *et al.* *Front. Immunol.* 2025.
       `PMID:41050667 <https://pubmed.ncbi.nlm.nih.gov/41050667/>`_
   * - TCRNET neighbourhood enrichment for antigen-specific TCR discovery
     - Neighborhood enrichment for the identification of antigen-specific
       T-cell receptors.
     - Lupyr KR *et al.* *Brief. Bioinform.* 2025.
       `PMID:40996146 <https://pubmed.ncbi.nlm.nih.gov/40996146/>`_
   * - TCREmp prototype embeddings for TCR sequence representation
     - TCREMP: A Bioinformatic Pipeline for Efficient Embedding of T-cell
       Receptor Sequences from Immune Repertoire and Single-cell Sequencing Data.
     - Kremlyakova Y *et al.* *J. Mol. Biol.* 2025.
       `PMID:40368275 <https://pubmed.ncbi.nlm.nih.gov/40368275/>`_
   * - TCRen structure-based prediction of TCR–epitope recognition
     - Structure-based prediction of T cell receptor recognition of unseen
       epitopes using TCRen.
     - Karnaukhov VK *et al.* *Nat. Comput. Sci.* 2024.
       `PMID:38987378 <https://pubmed.ncbi.nlm.nih.gov/38987378/>`_
   * - Clonal expansion analysis with deep single-cell repertoire profiling
     - Detecting T-cell clonal expansions and quantifying clone survival using
       deep profiling of immune repertoires.
     - Pavlova AV *et al.* *Front. Immunol.* 2024.
       `PMID:38633256 <https://pubmed.ncbi.nlm.nih.gov/38633256/>`_
   * - Regulatory T cell TCR repertoire convergence and tissue residence
     - Convergence, plasticity, and tissue residence of regulatory T cell
       response via TCR repertoire prism.
     - Nakonechnaya TO *et al.* *eLife* 2024.
       `PMID:38591522 <https://pubmed.ncbi.nlm.nih.gov/38591522/>`_
   * - Structural modeling of antigen-specific TCRs (template-based, large-scale)
     - Large-scale template-based structural modeling of T-cell receptors with
       known antigen specificity reveals complementarity features.
     - Shcherbinin DS *et al.* *Front. Immunol.* 2023.
       `PMID:37649481 <https://pubmed.ncbi.nlm.nih.gov/37649481/>`_
   * - Polyspecific CD8 T cells responding to multiple antigens (thymic repertoire)
     - Human thymopoiesis produces polyspecific CD8α/β T cells responding to
       multiple viral antigens.
     - Quiniou C *et al.* *eLife* 2023.
       `PMID:36995951 <https://pubmed.ncbi.nlm.nih.gov/36995951/>`_
   * - Antigen-driven TCR clonal groups in autoimmune spondyloarthritis synovial fluid
     - TCR repertoire profiling revealed antigen-driven CD8+ T cell clonal groups
       shared in synovial fluid of patients with spondyloarthritis.
     - Komech EA *et al.* *Front. Immunol.* 2022.
       `PMID:36325356 <https://pubmed.ncbi.nlm.nih.gov/36325356/>`_
   * - BCR repertoire memory persistence and antibody-secreting cell differentiation
     - Memory persistence and differentiation into antibody-secreting cells
       accompanied by positive selection in longitudinal BCR repertoires.
     - Mikelov AI *et al.* *eLife* 2022.
       `PMID:36107479 <https://pubmed.ncbi.nlm.nih.gov/36107479/>`_
   * - VDJdb SARS-CoV-2 update — antigen-specific TCR annotation resource
     - VDJdb in the pandemic era: a compendium of T cell receptors specific
       for SARS-CoV-2.
     - Goncharov M *et al.* *Nat. Methods* 2022.
       `PMID:35970936 <https://pubmed.ncbi.nlm.nih.gov/35970936/>`_
   * - Anti-PD-L1 response prediction using B-cell repertoire features
     - Accounting for B-cell Behavior and Sampling Bias Predicts Anti-PD-L1
       Response in Bladder Cancer.
     - Dyugay IA *et al.* *Cancer Immunol. Res.* 2022.
       `PMID:35013004 <https://pubmed.ncbi.nlm.nih.gov/35013004/>`_
   * - Ancestral diversity gap in published TCR sequencing studies
     - Ancestral diversity is limited in published T cell receptor sequencing studies.
     - Huang H *et al.* *Immunity* 2021.
       `PMID:34644550 <https://pubmed.ncbi.nlm.nih.gov/34644550/>`_
   * - Immunogenomics diversity and value of ancestral representation
     - Diversity in immunogenomics: the value and the challenge.
     - Peng K *et al.* *Nat. Methods* 2021.
       `PMID:34002093 <https://pubmed.ncbi.nlm.nih.gov/34002093/>`_
   * - TCR repertoire timestamp predicts CTLA-4 blockade response
     - A T cell repertoire timestamp is at the core of responsiveness to
       CTLA-4 blockade.
     - Philip M *et al.* *iScience* 2021.
       `PMID:33604527 <https://pubmed.ncbi.nlm.nih.gov/33604527/>`_
   * - Adaptive immunity organisation in long-lived rodent (Spalax galili)
     - Distinct organization of adaptive immunity in the long-lived rodent
       Spalax galili.
     - Izraelson M *et al.* *Nat. Aging* 2021.
       `PMID:37118630 <https://pubmed.ncbi.nlm.nih.gov/37118630/>`_
   * - SARS-CoV-2 public TCR repertoire and antigen-specific TCR annotation
     - SARS-CoV-2 Epitopes Are Recognized by a Public and Diverse Repertoire
       of Human T Cell Receptors.
     - Shomuradova AS *et al.* *Immunity* 2020.
       `PMID:33326767 <https://pubmed.ncbi.nlm.nih.gov/33326767/>`_
   * - T cell immune reconstitution tracking after αβ-depleted HSCT
     - T-cell tracking, safety, and effect of low-dose donor memory T-cell
       infusions after αβ T cell-depleted hematopoietic stem cell transplantation.
     - Blagov AV *et al.* *Bone Marrow Transplant.* 2020.
       `PMID:33203952 <https://pubmed.ncbi.nlm.nih.gov/33203952/>`_
   * - Benchmarking TCR repertoire profiling methods (systematic biases)
     - Benchmarking of T cell receptor repertoire profiling methods reveals
       large systematic biases.
     - Barennes P *et al.* *Nat. Biotechnol.* 2020.
       `PMID:32895550 <https://pubmed.ncbi.nlm.nih.gov/32895550/>`_
   * - CD4 T cell V-gene usage and biochemical features for influenza epitopes
     - CD4T Cells Recognize Conserved Influenza A Epitopes through Shared
       Patterns of V-Gene Usage and Complementary Biochemical Features.
     - Greenshields-Watson A *et al.* *Cell Rep.* 2020.
       `PMID:32668259 <https://pubmed.ncbi.nlm.nih.gov/32668259/>`_
   * - MHC-II allele shaping of CDR3 repertoires in naive CD4 T cell subsets
     - MHC-II alleles shape the CDR3 repertoires of conventional and
       regulatory naïve CD4 T cells.
     - Logunova NN *et al.* *Proc. Natl. Acad. Sci. USA* 2020.
       `PMID:32482872 <https://pubmed.ncbi.nlm.nih.gov/32482872/>`_
   * - TCRαβ chain pairing is nearly unconstrained (structural and sequencing data)
     - Comprehensive analysis of structural and sequencing data reveals almost
       unconstrained chain pairing in TCRαβ complex.
     - Shcherbinin DS *et al.* *PLoS Comput. Biol.* 2020.
       `PMID:32163410 <https://pubmed.ncbi.nlm.nih.gov/32163410/>`_
   * - B cell memory flexibility and resilience (CD27 subsets)
     - The Interplay between CD27 and CD27B Cells Ensures the Flexibility,
       Stability, and Resilience of Human B Cell Memory.
     - Grimsholm O *et al.* *Cell Rep.* 2020.
       `PMID:32130900 <https://pubmed.ncbi.nlm.nih.gov/32130900/>`_
   * - Benchmarking immunoinformatic tools for antibody repertoire analysis
     - Benchmarking immunoinformatic tools for the analysis of antibody
       repertoire sequences.
     - Smakaj E *et al.* *Bioinformatics* 2020.
       `PMID:31873728 <https://pubmed.ncbi.nlm.nih.gov/31873728/>`_
   * - VDJdb 2019 update — motif compendium and analysis infrastructure
     - VDJdb in 2019: database extension, new analysis infrastructure and a
       T-cell receptor motif compendium.
     - Bagaev DV *et al.* *Nucleic Acids Res.* 2020.
       `PMID:31588507 <https://pubmed.ncbi.nlm.nih.gov/31588507/>`_
   * - TCR–antigen database linking overview (immunoinformatics approaches)
     - An overview of immunoinformatics approaches and databases linking T cell
       receptor repertoires to their antigen specificity.
     - Zvyagin IV *et al.* *Immunogenetics* 2019.
       `PMID:31741011 <https://pubmed.ncbi.nlm.nih.gov/31741011/>`_
   * - Immune repertoire comparison summary statistics framework (sumrep)
     - sumrep: A Summary Statistic Framework for Immune Receptor Repertoire
       Comparison and Model Validation.
     - Olson BJ *et al.* *Front. Immunol.* 2019.
       `PMID:31736960 <https://pubmed.ncbi.nlm.nih.gov/31736960/>`_
   * - Tumour immunoglobulin isotypes predict lung adenocarcinoma survival
     - Intratumoral immunoglobulin isotypes predict survival in lung
       adenocarcinoma subtypes.
     - Isaeva OI *et al.* *J. Immunother. Cancer* 2019.
       `PMID:31665076 <https://pubmed.ncbi.nlm.nih.gov/31665076/>`_
   * - Antigen-specificity annotation framework for high-throughput TCR studies
     - A Framework for Annotation of Antigen Specificities in
       High-Throughput T-Cell Repertoire Sequencing Studies.
     - Pogorelyy MV, Shugay M (2019) *Front. Immunol.* 10:2159.
       `PMID:31616409 <https://pubmed.ncbi.nlm.nih.gov/31616409/>`_
   * - ALICE enrichment — antigen-driven TCR cluster detection from snapshots
     - Detecting T cell receptors involved in immune responses from single
       repertoire snapshots.
     - Pogorelyy MV *et al.* *PLoS Biol.* 2019.
       `PMID:31194732 <https://pubmed.ncbi.nlm.nih.gov/31194732/>`_
   * - B cell repertoire comparison across vaccine response cohorts
     - Comparative Analysis of B-Cell Receptor Repertoires Induced by Live
       Yellow Fever Vaccine in Young and Middle-Age Donors.
     - Davydov AN *et al.* *Front. Immunol.* 2018.
       `PMID:30356675 <https://pubmed.ncbi.nlm.nih.gov/30356675/>`_
   * - High-resolution Tfh/Tfr repertoire; bystander activation analysis
     - High-resolution repertoire analysis reveals a major bystander activation
       of Tfh and Tfr cells.
     - Ritvo P-G *et al.* *Proc. Natl. Acad. Sci. USA* 2018.
       `PMID:30158170 <https://pubmed.ncbi.nlm.nih.gov/30158170/>`_
   * - Pre-immune antigen-specific TCR landscape; Pgen context for ALICE
     - Exploring the pre-immune landscape of antigen-specific T cells.
     - Pogorelyy MV *et al.* *Genome Med.* 2018.
       `PMID:30144804 <https://pubmed.ncbi.nlm.nih.gov/30144804/>`_
   * - Naive T cell repertoire aging dynamics (longitudinal cohort)
     - The Changing Landscape of Naive T Cell Receptor Repertoire With
       Human Aging.
     - Egorov ES *et al.* *Front. Immunol.* 2018.
       `PMID:30087674 <https://pubmed.ncbi.nlm.nih.gov/30087674/>`_
   * - Antigen-specific TCR motifs in ankylosing spondylitis synovial fluid
     - CD8+ T cells with characteristic T cell receptor beta motif are detected
       in blood and expanded in synovial fluid of ankylosing spondylitis patients.
     - Komech EA *et al.* *Rheumatology (Oxford)* 2018.
       `PMID:29481668 <https://pubmed.ncbi.nlm.nih.gov/29481668/>`_
   * - VDJdb curated antigen-specific TCR database (original)
     - VDJdb: a curated database of T-cell receptor sequences with known
       antigen specificity.
     - Shugay M *et al.* *Nucleic Acids Res.* 2018.
       `PMID:28977646 <https://pubmed.ncbi.nlm.nih.gov/28977646/>`_
   * - Comparative murine TCR repertoire analysis methodology
     - Comparative analysis of murine T-cell receptor repertoires.
     - Izraelson M *et al.* *Immunology* 2017.
       `PMID:29080364 <https://pubmed.ncbi.nlm.nih.gov/29080364/>`_
   * - RNA-seq-based antigen receptor repertoire profiling
     - Antigen receptor repertoire profiling from RNA-seq data.
     - Bolotin DA *et al.* *Nat. Biotechnol.* 2017.
       `PMID:29020005 <https://pubmed.ncbi.nlm.nih.gov/29020005/>`_
   * - Unique molecular barcode library preparation (NOPE method)
     - Application of nonsense-mediated primer exclusion (NOPE) for preparation
       of unique molecular barcoded libraries.
     - Shagin DA *et al.* *BMC Genomics* 2017.
       `PMID:28583065 <https://pubmed.ncbi.nlm.nih.gov/28583065/>`_
   * - PCR error quantification for high-accuracy repertoire sequencing
     - A high-throughput assay for quantitative measurement of PCR errors.
     - Shagin DA *et al.* *Sci. Rep.* 2017.
       `PMID:28578414 <https://pubmed.ncbi.nlm.nih.gov/28578414/>`_
   * - Molecular-barcoded targeted resequencing pipeline (MAGERI)
     - MAGERI: Computational pipeline for molecular-barcoded targeted
       resequencing.
     - Shugay M *et al.* *PLoS Comput. Biol.* 2017.
       `PMID:28475621 <https://pubmed.ncbi.nlm.nih.gov/28475621/>`_
   * - T cell immune reconstitution tracking after αβ/CD19-depleted HSCT
     - Tracking T-cell immune reconstitution after TCRαβ/CD19-depleted
       hematopoietic cells transplantation in children.
     - Zvyagin IV *et al.* *Leukemia* 2016.
       `PMID:27811849 <https://pubmed.ncbi.nlm.nih.gov/27811849/>`_
   * - High-quality immunoglobulin profiling with unique molecular barcoding
     - High-quality full-length immunoglobulin profiling with unique molecular
       barcoding.
     - Turchaninova MA *et al.* *Nat. Protoc.* 2016.
       `PMID:27490633 <https://pubmed.ncbi.nlm.nih.gov/27490633/>`_
   * - VDJviz — versatile browser for immunogenomics data visualisation
     - VDJviz: a versatile browser for immunogenomics data.
     - Bagaev DV *et al.* *BMC Genomics* 2016.
       `PMID:27297497 <https://pubmed.ncbi.nlm.nih.gov/27297497/>`_
   * - Single-cell TCR analysis in Sjögren's syndrome glandular T cells
     - Single-cell analysis of glandular T cell receptors in Sjögren's syndrome.
     - Joachims ML *et al.* *JCI Insight* 2016.
       `PMID:27358913 <https://pubmed.ncbi.nlm.nih.gov/27358913/>`_
   * - T cell repertoire dynamics across 79 donors (cord blood to centenarians)
     - Dynamics of Individual T Cell Repertoires: From Cord Blood to Centenarians.
     - Britanova OV *et al.* *J. Immunol.* 2016.
       `PMID:27183615 <https://pubmed.ncbi.nlm.nih.gov/27183615/>`_
   * - Regulatory T cell repertoire expansion mechanism (Treg context)
     - A mechanism for expansion of regulatory T-cell repertoire and its role
       in self-tolerance.
     - Feng Y *et al.* *Nature* 2015.
       `PMID:26605529 <https://pubmed.ncbi.nlm.nih.gov/26605529/>`_
   * - Diversity metrics and post-analysis toolkit (VDJtools)
     - VDJtools: Unifying Post-analysis of T Cell Receptor Repertoires.
     - Shugay M *et al.* *PLoS Comput. Biol.* 2015.
       `PMID:26606115 <https://pubmed.ncbi.nlm.nih.gov/26606115/>`_
   * - Rare T cell population sequencing strategies
     - Sequencing rare T-cell populations.
     - Shugay M *et al.* *Oncotarget* 2015.
       `PMID:26588057 <https://pubmed.ncbi.nlm.nih.gov/26588057/>`_
   * - Foxp3+ Treg TCR subsets defined by CD39 and CD45RO expression
     - TCR usage, gene expression and function of two distinct FOXP3(+)Treg
       subsets within CD4(+)CD25(hi) T cells identified by expression of
       CD39 and CD45RO.
     - Ye J *et al.* *Immunol. Cell Biol.* 2015.
       `PMID:26467610 <https://pubmed.ncbi.nlm.nih.gov/26467610/>`_
   * - Quantitative immune repertoire profiling with unique molecular identifiers
     - Quantitative profiling of immune repertoires for minor lymphocyte counts
       using unique molecular identifiers.
     - Egorov ES *et al.* *J. Immunol.* 2015.
       `PMID:25957172 <https://pubmed.ncbi.nlm.nih.gov/25957172/>`_
   * - tcR — R package for TCR repertoire data analysis
     - tcR: an R package for T cell receptor repertoire advanced data analysis.
     - Nazarov VI *et al.* *BMC Bioinformatics* 2015.
       `PMID:26017500 <https://pubmed.ncbi.nlm.nih.gov/26017500/>`_
   * - Error-free immune repertoire profiling methodology
     - Towards error-free profiling of immune repertoires.
     - Shugay M *et al.* *Nat. Methods* 2014.
       `PMID:24793455 <https://pubmed.ncbi.nlm.nih.gov/24793455/>`_
   * - Identical twins' TCR repertoires (deep sequencing, clonotype sharing)
     - Distinctive properties of identical twins' TCR repertoires revealed by
       high-throughput sequencing.
     - Zvyagin IV *et al.* *Proc. Natl. Acad. Sci. USA* 2014.
       `PMID:24711416 <https://pubmed.ncbi.nlm.nih.gov/24711416/>`_
   * - Age-related TCR diversity decrease (deep and normalised profiling)
     - Age-related decrease in TCR repertoire diversity measured with deep and
       normalized sequence profiling.
     - Britanova OV *et al.* *J. Immunol.* 2014.
       `PMID:24510963 <https://pubmed.ncbi.nlm.nih.gov/24510963/>`_
   * - Massive individual TCR beta repertoire overlap (diversity baseline)
     - Huge Overlap of Individual TCR Beta Repertoires.
     - Shugay M *et al.* *Front. Immunol.* 2013.
       `PMID:24400005 <https://pubmed.ncbi.nlm.nih.gov/24400005/>`_
   * - Mother–child TCR repertoire sharing (deep profiling)
     - Mother and child T cell receptor repertoires: deep profiling study.
     - Putintseva EV *et al.* *Front. Immunol.* 2013.
       `PMID:24400004 <https://pubmed.ncbi.nlm.nih.gov/24400004/>`_
   * - TCRαβ chain pairing by emulsion PCR (paired-chain sequencing)
     - Pairing of T-cell receptor chains via emulsion PCR.
     - Turchaninova MA *et al.* *Eur. J. Immunol.* 2013.
       `PMID:23696157 <https://pubmed.ncbi.nlm.nih.gov/23696157/>`_
