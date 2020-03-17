import pytest

import scanpy as sc
import scanpy.external as sce

pytest.importorskip("doubletdetection")


def test_doublet_detection():
    adata = sc.datasets.pbmc3k()
    adata.layers['raw'] = adata.X
    sce.external.tl.doublet_detection(adata)
    assert 'doublet_detection_labels' in adata.obs, "Run DoubletDetection Error!"
