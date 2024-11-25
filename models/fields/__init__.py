from .classifier import SpatialClassifierVN, SimpleEdgePredictor, FragmentPosDecoder


def get_field_vn(config, num_classes, in_sca, in_vec, **kwargs):
    if config.name == 'classifier':
        return SpatialClassifierVN(
            num_classes = num_classes,
            in_vec = in_vec,
            in_sca = in_sca,
            num_filters = [config.num_filters, config.num_filters_vec],
            edge_channels = config.edge_channels,
            cutoff = config.cutoff,
            **kwargs
        )
    else:
        raise NotImplementedError('Unknown field: %s' % config.name)
