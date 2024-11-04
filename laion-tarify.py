import argparse
from make_tar_utils import tardir, packup
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=str,
    default="CMU-MOSI/data",
    help="input folder, expecting subdirectory like train, valid or test",
)
parser.add_argument(
    "--output",
    type=str,
    default="tar/cmu-mosi/",
    help="output, generating tar files at output/dataclass/filename_{}.tar",
)
parser.add_argument(
    "--filename",
    type=str,
    default="",
    help="the filename of the tar, generating tar files at output/dataclass/filename_{}.tar",
)
parser.add_argument(
    "--dataclass", type=str, default="all", help="train or test or valid or all"
)
parser.add_argument(
    "--num_element", type=int, default=2048, help="pairs of (audio, text) to be included in a single tar"
)
parser.add_argument(
    "--start_idx", type=int, default=0, help="start index of the tar"
)
args = parser.parse_args()


if __name__ == "__main__":
    if args.dataclass == "all":
        for x in ["train", "valid", "test"]:
            packup(args.input, args.output,  args.filename,  x,  args.num_element)
    elif args.dataclass == "none":
        os.makedirs(args.output, exist_ok=True)
        tardir(
            args.input,
            args.output,
            args.num_element,
            start_idx=0,
            delete_file=False,
        )
    else:  # if dataclass is in other name
        packup(args.input, args.output,  args.filename,  args.dataclass,  args.num_element)