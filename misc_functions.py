#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:56:41 2018

@author: Alexandre
"""

import string
import numpy as np




def ranking_format(string_rank):
    # je suppose que tout est toujours format comme iris
    L_letters = string_rank.split('>')
    L_ranking = [string.lowercase.index(letter) for letter in L_letters]
    return L_ranking

def ranking_format_sorted(string_rank):
    # je suppose que tout est toujours format comme iris
    L_letters = string_rank.split('>')
    L_ranking = np.argsort([string.lowercase.index(letter) for letter in L_letters]).tolist()

    return L_ranking


def encode_partial(string_rank):


    L_letters = string_rank.split('>')
    n_letters = np.sum([len(i) for i in L_letters])
    partial_rank_list = np.zeros(n_letters)


    # conversion to partial ranking
    rank_level = 0
    for let_list in L_letters:

        if len(let_list)>1: # items with equal rank
            for letter in let_list:
                rank_letter = string.lowercase.index(letter)
                partial_rank_list[rank_letter] = rank_level
        else:
            rank_letter = string.lowercase.index(let_list)
            partial_rank_list[rank_letter] = rank_level
        rank_level +=1
    return partial_rank_list

def miss_label_30_top(sigma):
    return miss_label_top(sigma,0.7)

def miss_label_60_top(sigma):
    return miss_label_top(sigma,0.4)

def miss_label_top(sigma,p):
    n = len(sigma)
    sigma_top = np.copy(sigma)
    p_kept = int(np.round(p*(n-1)))
    sigma_top[sigma_top >= p_kept] = p_kept
    return sigma_top.tolist()