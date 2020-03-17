from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse.base import spmatrix

from ... import logging as logg


def doublet_detection(
    adata: AnnData,
    use_raw: Optional[bool] = True,
    layer: Optional[str] = None,
    boost_rate: float = 0.25,
    n_components: int = 30,
    n_top_var_genes: int = 10000,
    new_lib_as: Optional[int] = None,
    replace: bool = False,
    n_iters: int = 25,
    normalizer: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    prune: bool = True,
    p_thresh: float = 1e-7,
    voter_thresh: float = 0.9,
    copy: bool = False,
    **phenograph_kwargs,
):
    """\
    DoubletDetection [Gayoso18]_.

    DoubletDetection is a Python3 package to detect doublets (technical errors) in
    single-cell RNA-seq count matrices.

    As input, the tool **requires** an array-like raw counts matrix of cells `x`
    genes of scRNA-seq.

    The return is a 1-dimensional `numpy.ndarray` with the value 1 representing a
    detected doublet, 0 a singlet, and `np.nan` as ambiguous cell.

    The classifier works best when:

    - There are several cell types present in the data.
    - It is applied separately to each run in an aggregated count matrix.

    .. note::
       More information and bug reports `here
       <https://github.com/dpeerlab/DoubletDetection>`__.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_raw
        Use `.raw` attribute of `adata`. If `None`, defaults to `True` if `layer`
        isn't provided and `adata.raw` is present.
    layer
        Name of the AnnData object layer to be used. By default adata.raw.X is
        used. If `use_raw=False` is set, then `layer` is used. `layer` takes
        precedence over `use_raw`
    boost_rate
        Proportion of cell population size to produce as synthetic doublets.
    n_components
        Number of principal components used for clustering.
    n_top_var_genes
        Number of highest variance genes to use; other genes discarded.
        Will use all genes when zero.
    new_lib_as
        Method to use in choosing library size for synthetic doublets. Defaults to
        None which makes synthetic doublets the exact addition of its parents;
        alternative is `new_lib_as=np.max`.
    replace
        If False, a cell will be selected as a synthetic doublet's parent no more
        than once.
    n_iters
        Number of fit operations from which to collect p-values.
    normalizer
        Method to normalize raw_counts. Defaults to `normalize_counts`, included in
        this package. Note: To use `normalize_counts` with its pseudo count parameter
        changed from the default 0.1 value to some positive float `new_var`, use:
        normalizer=lambda counts: doubletdetection.normalize_counts(counts,
        pseudocount=new_var)
    random_state
        Passed to PCA and `np.random.seed`. NOTE: In phenograph,
        this parameter is passed to `seed`- Leiden initialization of the optimization,
        when choosing `clustering_algo='leiden'`.
    prune
        PhenoGraph parameter; Whether to symmetrize by taking the average (
        prune=False) or product (prune=True) between the graph and its transpose
    p_thresh
        Passed to :func:`~doubletdetection.BoostClassifier.predict`. The hypergeometric
        test p-value threshold that determines per iteration doublet calls.
    voter_thresh
        Passed to :func:`~doubletdetection.BoostClassifier.predict`. Fraction of
        iterations a cell must be called a doublet.
    copy
        Return a copy or write to `adata`.
    phenograph_kwargs
        Any further arguments to pass to `phenograph.cluster`.

    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields:

    doublet_detection_labels : `numpy.ndarray` (`adata.obs['doublet_detection_labels']`)
        Integer array of doublets detected.

    Example
    -------

    >>> from anndata import AnnData
    >>> import scanpy as sc
    >>> import scanpy.external as sce

    >>> adata = sc.datasets.pbmc3k()

    Since the tool works on `raw` data only, `adata` must have one of `adata.raw` or a
    layer present, with raw counts, that can be read and processed. As such,

    >>> adata.raw = adata

    or,

    >>> adata.layers['raw'] = adata.X

    conditional on that `adata.X` is raw and not normalized data.

    **Reading from `adata.raw`**

    >>> sce.external.tl.doublet_detection(adata)

    With PhenoGraph version >=1.5.3, a new parameter allows to choose between Louvain
    or Leiden algorithms for community detection, by passing `clustering_algo`,
    for example:

    >>> sce.external.tl.doublet_detection(adata, clustering_algo='leiden')

    **Reading from a Layer**

    >>> sce.external.tl.doublet_detection(
    ...     adata,
    ...     use_raw=False,
    ...     layer='raw',
    ... )

    """
    try:
        import doubletdetection
    except ImportError:
        raise ImportError(
            "\nplease install doubletdetection:\n\n \tpip install doubletdetection"
        )

    phenograph_kwargs["prune"] = prune
    if random_state:
        phenograph_kwargs["seed"] = random_state

    if use_raw is None:
        # check if adata.raw is set
        use_raw = layer is None and adata.raw is not None
    if use_raw and layer is not None:
        raise ValueError(
            "Cannot use both a layer and the raw representation. Was passed:"
            f"use_raw={use_raw}, layer={layer}."
        )
    if adata.raw is None and use_raw:
        raise ValueError(
            "`use_raw` is set to True but AnnData object does not have raw. "
            "Please check."
        )
    if layer is None and not use_raw:
        raise ValueError("`use_raw` is set to False while `layer` is set not set.")

    if use_raw:
        print("Doublet Detection using `adata.raw.X` ....")
        logg.info("Doublet Detection using `adata.raw.X` ....")
        raw_counts = adata.raw.X
    else:
        assert (
            layer in adata.layers
        ), "Layer key `'{}'` not found in " "adata.layers".format(layer)

        print(f"Doublet Detection using `adata.layers['{layer}'']` ....")
        logg.info(f"Doublet Detection using `adata.layers['{layer}'']` ....")
        raw_counts = adata.layers[layer]

    if isinstance(raw_counts, spmatrix):
        raw_counts = raw_counts.toarray()

    clf = doubletdetection.BoostClassifier(
        boost_rate=boost_rate,
        n_components=n_components,
        n_top_var_genes=n_top_var_genes,
        new_lib_as=new_lib_as,
        replace=replace,
        phenograph_parameters=phenograph_kwargs,
        n_iters=n_iters,
        normalizer=normalizer,
        random_state=random_state,
    )

    labels = clf.fit(raw_counts=raw_counts).predict(
        p_thresh=p_thresh, voter_thresh=voter_thresh
    )

    if copy:
        return labels
    else:
        adata.obs["doublet_detection_labels"] = pd.Categorical(
            labels, categories=[0, 1]
        )
