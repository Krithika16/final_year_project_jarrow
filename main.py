from augpolicies.core.util.parse_args import get_args
from augpolicies.core.util.logging import get_logger


args = get_args()
logger = get_logger(args)

try:
    if args.hpc:
        if args.rank:
            logger.info("starting aug_comparison_ranking...")
            import augpolicies.hypothesis_testing.aug_comparison_ranking as acr
            acr.main(args)
        elif args.vis:
            logger.info("starting aug_comparison_analysis...")
            import augpolicies.hypothesis_testing.aug_comparison_analysis as aca
            aca.main(args)
        else:
            logger.info("starting aug_comparison...")
            import augpolicies.hypothesis_testing.aug_comparison as ac
            ac.main(args)
    elif args.hpe:
        # if args.rank:
        #     import augpolicies.hypothesis_testing.augmentation_at_end_ranking
        if args.vis:
            logger.info("starting augmentation_at_end_analysis...")
            import augpolicies.hypothesis_testing.augmentation_at_end_analysis as aaea
            aaea.main(args)
        else:
            logger.info("starting augmentation_at_end...")
            import augpolicies.hypothesis_testing.augmentation_at_end as aae
            aae.main(args)
except (SystemExit, KeyboardInterrupt):
    logger.info("SystemExit or KeyboardInterrupt.")
    raise
except Exception as exception:
    logger.error('Exception:', exc_info=True)
else:
    logger.info("Complete.")
