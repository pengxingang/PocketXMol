from .cftfm import CFTransformerEncoder, CFTransformerEncoderVN


def get_encoder_vn(config, **kwargs):
    if config.name == 'cftfm':
        return CFTransformerEncoder(
            hidden_channels = config.hidden_dim,
            edge_channels = config.edge_channels,
            num_interactions = config.num_interactions,
            k = config.knn,
            cutoff = config.cutoff,
            **kwargs,
        )
    elif config.name == 'cftfm_vn':
        return CFTransformerEncoderVN(
            hidden_channels = [config.hidden_channels, config.hidden_channels_vec],
            edge_channels = config.edge_channels,
            num_interactions = config.num_interactions,
            k = config.knn,
            cutoff = config.cutoff,
            **kwargs,
        )
    else:
        raise NotImplementedError('Unknown encoder: %s' % config.name)
