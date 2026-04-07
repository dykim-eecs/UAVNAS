#!*****************************************************************************************
#! [UAV-NAS Project] CONFIDENTIAL & PROPRIETARY                                            *
#! Copyright (c) 2026 Doh-Yon Kim (First Author), All Rights Reserved.                     *
#!                                                                                         *
#! NOTICE: This source code is associated with the following academic publication:         *
#! "UAV-NAS: UAV Identification on FPGAs via Neural Architecture Search"                   *
#! Accepted for publication in IEEE Transactions on Industrial Informatics (TII).          *
#!                                                                                         *
#! All information contained herein is, and remains the property of the authors.           *
#! The intellectual and technical concepts contained herein are protected by copyright     *
#! law and international treaties. Unauthorized copying, modification, or distribution     *
#! of this file, via any medium, is strictly prohibited.                                   *
#!                                                                                         *
#!*****************************************************************************************
#
#* [ PAPER METADATA ]
#  - Title   : UAV-NAS: UAV Identification on FPGAs via Neural Architecture Search
#  - Journal : IEEE Transactions on Industrial Informatics (TII)
#
#* [ AUTHOR INFORMATION ]
#  - First Author : Kim, Doh-Yon
#
#* [ FILE DESCRIPTION ]
#  - Module Name : genotypes
#  - Function    : Genotype structure and the NAS-discovered architecture used for inference
#
#

"""Genotype container and the final architecture chosen by the DARTS search."""

from collections import namedtuple

# Cell genotype: (op, input_index) lists for normal/reduce cells, plus concat indices
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# Best genotype produced by the search; replace this with new search results when re-running
NASGenotype = Genotype(
    normal=[
        ('sep_conv_5x5', 1), ('skip_connect', 0),
        ('sep_conv_3x3', 2), ('dil_conv_3x3', 0),
        ('avg_pool_3x3', 0), ('dil_conv_3x3', 2),
        ('sep_conv_3x3', 3), ('max_pool_3x3', 1),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('dil_conv_5x5', 1), ('sep_conv_5x5', 0),
        ('skip_connect', 1), ('sep_conv_5x5', 2),
        ('dil_conv_3x3', 2), ('sep_conv_5x5', 3),
        ('dil_conv_5x5', 4), ('skip_connect', 1),
    ],
    reduce_concat=[2, 3, 4, 5],
)
