import argparse
import os

from core import config, data, radiomics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--save-path", default=config.rad_feats_save)
    args = parser.parse_args()

    hgg_dir = os.path.join(args.root, "HGG")
    case_dirs = data.list_case_dirs(hgg_dir)

    feats, keys, cases = radiomics.precompute_hgg_radiomics(
        case_dirs,
        load_case_fn=data.load_case,
        save_path=args.save_path,
    )

    print(
        f"saved {feats.shape[0]} cases with "
        f"{feats.shape[1]} features to {args.save_path}"
    )


if __name__ == "__main__":
    main()
