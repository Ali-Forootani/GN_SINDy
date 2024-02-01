""" Useful tools, such as combining string matrix product and computing the terms in the library """
from itertools import product, combinations, chain
import torch
import torch.nn as nn

import numpy as np
import torch
import sys
import os
import scipy.io as sio
import shutil

cwd = os.getcwd()
sys.path.append(cwd)



def string_matmul(list_1, list_2):
    """Matrix multiplication with strings."""
    prod = [element[0] + element[1] for element in product(list_1, list_2)]
    return prod


def terms_definition(poly_list, deriv_list):
    """Calculates which terms are in the library."""
    if len(poly_list) == 1:
        theta = string_matmul(
            poly_list[0], deriv_list[0]
        )  # If we have a single output, we simply calculate and flatten matrix product between polynomials and derivatives to get library
    else:
        theta_uv = list(
            chain.from_iterable(
                [string_matmul(u, v) for u, v in combinations(poly_list, 2)]
            )
        )  # calculate all unique combinations between polynomials
        theta_dudv = list(
            chain.from_iterable(
                [string_matmul(du, dv)[1:] for du, dv in combinations(deriv_list, 2)]
            )
        )  # calculate all unique combinations of derivatives
        theta_udu = list(
            chain.from_iterable(
                [
                    string_matmul(u[1:], du[1:])
                    for u, du in product(poly_list, deriv_list)
                ]
            )
        )  # calculate all unique combinations of derivatives
        theta = theta_uv + theta_dudv + theta_udu
    return theta




def create_or_reset_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # If it exists, remove it
        try:
            #os.rmdir(directory_path)
            shutil.rmtree(directory_path)
            print(f"Directory '{directory_path}' already exist so it is removed.")
            os.rmdir(directory_path)
            print(f"Directory '{directory_path}' is created.")
        except OSError as e:
            print(f"Error removing directory '{directory_path}': {e}")
            return

    # Create the directory
    try:
        os.mkdir(directory_path)
        print(f"Directory '{directory_path}' created.")
    except OSError as e:
        print(f"Error creating directory '{directory_path}': {e}")