import argparse

parser = argparse.ArgumentParser(description='Script to run augpolicies')
parser.add_argument('--hpc', action='store_true', help='Hypothesis testing aug comparison')
parser.add_argument('--hpe', action='store_true', help='Hypothesis testing aug at end')
parser.add_argument('--vis', action='store_true', help='Hypothesis testing visualise result')
args = parser.parse_args()

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
