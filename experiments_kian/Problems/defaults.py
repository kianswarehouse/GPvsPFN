import gpplus
import torch

SF_kernel = None
SF_mean = None
SF_likelihood = None

MF_mean = gpplus.means.MultiMean
MF_likelihood = gpplus.likelihoods.MultiLikelihood
def MF_kernel(
    cont_cols=None,
    cat_cols=None,
    source_cols=None,
    cont_kernel=None,
    cat_kernel=None,
    source_kernel=None,
    cat_encoder=None,
    source_encoder=None,
    z_dim=2,
    fix_lengthscale_cat=False,
    fix_lengthscale_source=False,
    **kwargs
):
    """Factory function that creates and wraps the MF kernel."""
    return gpplus.kernels.LogScaleKernel(
        gpplus.kernels.MVMFKernel(
            cont_cols=cont_cols,
            cat_cols=cat_cols,
            source_cols=source_cols,
            cont_kernel=cont_kernel,
            cat_kernel=cat_kernel,
            source_kernel=source_kernel,
            cat_encoder=cat_encoder,
            source_encoder=source_encoder,
            z_dim=z_dim,
            fix_lengthscale_cat=fix_lengthscale_cat,
            fix_lengthscale_source=fix_lengthscale_source,
            **kwargs
        )
    )

