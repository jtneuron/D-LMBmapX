from model.registration.voxelmorph.voxelmorph import VxmDense


def create_model(
        spatial_dims,
        int_steps,
        use_probs,
        src_feats,
        trg_feats):
    enc_nf, dec_nf = channel_defaults()
    return VxmDense(
        spatial_dims=spatial_dims,
        nb_unet_features=[enc_nf, dec_nf],
        int_steps=int_steps,
        use_probs=use_probs,
        src_feats=src_feats,
        trg_feats=trg_feats,
    )


def channel_defaults():
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    return enc_nf, dec_nf


def voxelmorph_defaults():
    config = dict(
        spatial_dims=2,
        int_steps=0,
        use_probs=False,
        src_feats=1,
        trg_feats=1,
    )
    return config


def voxelmorph_diff_defaults():
    config = dict(
        spatial_dims=2,
        int_steps=7,
        use_probs=False,
        src_feats=1,
        trg_feats=1,
    )
    return config


def voxelmorph_prob_defaults():
    config = dict(
        spatial_dims=2,
        int_steps=0,
        use_probs=True,
        src_feats=1,
        trg_feats=1,
    )
    return config


def voxelMorph_diff_prob():
    config = dict(
        spatial_dims=2,
        int_steps=7,
        use_probs=True,
        src_feats=1,
        trg_feats=1,
    )
    return config
