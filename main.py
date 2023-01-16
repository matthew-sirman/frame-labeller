#! /usr/bin/python3
"""
Main module.

Entry point to the program
"""
import config as cfg
from train import train_model
from predict import create_resource


def main():
    """Entry point to program."""
    if cfg.GENERATE_RESOURCE:
        create_resource()
    else:
        train_model()


if __name__ == "__main__":
    main()
