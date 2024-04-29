architecture = [
    {'type': 'Dense', 'layer': 'IBL', 'n_repeats': 8, 'kernel_size': 5},
    {'type': 'Transition', 'out_channels': 64},
    {'type': 'Dense', 'layer': 'IBL', 'n_repeats': 8, 'kernel_size': 5},
    {'type': 'Transition', 'out_channels': 64},
    {'type': 'Dense', 'layer': 'AAIBL', 'n_repeats': 6, 'kernel_size': 3, 'n_heads': 1, 'size': 56},
    {'type': 'Transition', 'out_channels': 64},
    {'type': 'Dense', 'layer': 'AAIBL', 'n_repeats': 8, 'kernel_size': 3, 'n_heads': 4, 'size': 28},
    {'type': 'Transition', 'out_channels': 64},
    {'type': 'Dense', 'layer': 'AAIBL', 'n_repeats': 10, 'kernel_size': 3, 'n_heads': 4, 'size': 14},
    {'type': 'Transition', 'out_channels': 64},
    {'type': 'Dense', 'layer': 'AAIBL', 'n_repeats': 12, 'kernel_size': 3, 'n_heads': 4, 'size': 7},
    {'type': 'Transition', 'out_channels': 128},
    {'type': 'Dense', 'layer': 'AAIBL', 'n_repeats': 14, 'kernel_size': 3, 'n_heads': 4, 'size': 4},
    {'type': 'Transition', 'out_channels': 128},
    {'type': 'Dense', 'layer': 'AAIBL', 'n_repeats': 32, 'kernel_size': 2, 'n_heads': 4, 'size': 2},
    {'type': 'AAIBL', 'out_channels': 100, 'kernel_size': 2, 'n_heads': 10, 'size': 2},
    {'type': 'AvgPool', 'kernel_size': 2, 'stride': 2},
    {'type': 'Conv', 'kernel_size': 1, 'out_channels': 42} # 42 = (number of points on hand) * 2
]