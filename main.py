from augpolicies.core.util.parse_args import get_args
from augpolicies.core.util.logging import get_logger


args = get_args()
logger = get_logger(args)

try:
    if args.hpc:
        if args.rank:
            import augpolicies.hypothesis_testing.aug_comparison_ranking
        elif args.vis:
            import augpolicies.hypothesis_testing.aug_comparison_analysis
        else:
            import augpolicies.hypothesis_testing.aug_comparison
    elif args.hpe:
        # if args.rank:
        #     import augpolicies.hypothesis_testing.augmentation_at_end_ranking
        if args.vis:
            import augpolicies.hypothesis_testing.augmentation_at_end_analysis
        else:
            import augpolicies.hypothesis_testing.augmentation_at_end
except (SystemExit, KeyboardInterrupt):
    raise
except Exception as exception:
    logger.error('Failed to open file', exc_info=True)
