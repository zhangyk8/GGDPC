#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Aug 26, 2026

Description: Synthesize the outputs from our simulation studies
"""

import numpy as np
import pandas as pd

#=======================================================================================#

B = 1000
all_results = []
for n in [500, 1000, 2000, 5000, 8000]:
    # Load the results from each job
    results_list = []
    for job_id in range(1, B + 1):
        file_path = f"./Results/results_GMM_{n}_samples_job_{job_id}_new.csv"
        try:
            df = pd.read_csv(file_path)
            df['n_samples'] = n  # Add a column for sample size
            results_list.append(df)
        except FileNotFoundError:
            print(f"File not found: {file_path}")

    # Concatenate all results into a single DataFrame
    if results_list:
        all_results.append(pd.concat(results_list, ignore_index=True))

# Combine all sample sizes into a single DataFrame
if all_results:
    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_csv("./Syn_Results/GMM_results_new.csv", index=False)