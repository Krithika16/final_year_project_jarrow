from augpolicies.core.util.parse_args import get_args
args = get_args()

if args.hpc:
    if args.vis:
        import augpolicies.hypothesis_testing.aug_comparison_analysis
    else:
        import augpolicies.hypothesis_testing.aug_comparison
elif args.hpe:
    if args.vis:
        import augpolicies.hypothesis_testing.augmentation_at_end_analysis
    else:
        import augpolicies.hypothesis_testing.augmentation_at_end
