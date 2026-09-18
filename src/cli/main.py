from __future__ import annotations

import argparse

from pipelines import run_pipeline


def main():
    parser=argparse.ArgumentParser(prog="banknifty")
    parser.add_argument("command",choices=["pipeline"])
    args=parser.parse_args()
    if args.command=="pipeline":
        run_pipeline()


if __name__=="__main__":
    main()
