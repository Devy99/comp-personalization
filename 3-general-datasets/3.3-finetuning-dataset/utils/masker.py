from __future__ import annotations
from abc import ABC, abstractmethod
import os, csv, random

random.seed(230923)

class Masker(ABC):

    def __init__(self, methods: list, idx_to_mask: list = None, idx_blacklist: list = None, generate_random_if_none: bool = True,\
                 extra_columns_header: list = list(), extra_columns_list: list = list(), \
                 sep: str = '\n', max_masking: int = -1,  mask_symbol: str = '<extra_id_0>'):
        self._methods = methods
        self._idx_to_mask = idx_to_mask if idx_to_mask else [None for _ in range(len(self._methods))]
        self._idx_blacklist = idx_blacklist if idx_blacklist else [None for _ in range(len(self._methods))]
        self._generate_random_if_none = generate_random_if_none
        self._extra_columns_header = extra_columns_header
        self._extra_columns_list = extra_columns_list
        self._sep = sep
        self._max_masking = max_masking
        self._mask_symbol = mask_symbol

        self._extra_columns = [list() for _ in range(len(extra_columns_header))] if extra_columns_header else None
        self._masked_codes, self._masks = list(), list()
        self._masked_indexes = list()

    @abstractmethod
    def mask(self) -> list:
        pass

    def export(self, out_fp: str) -> None:
        # Init CSV if not exists
        if not os.path.isfile(out_fp):
            with open(out_fp, 'w', newline='') as f:
                writer = csv.writer(f)
                header = self._extra_columns_header + ['masked_method', 'mask']
                writer.writerow(header)

        # Update with the input rows
        with open(out_fp, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if self._extra_columns:
              extra_columns = [list(extra_cols) for extra_cols in zip(*self._extra_columns)]
              data_rows = [cols + [masked, mask] for cols, masked, mask in zip(extra_columns, self._masked_codes, self._masks)]
            else: 
              data_rows = [[masked, mask] for masked, mask in zip(self._masked_codes, self._masks)]
            writer.writerows(data_rows)

    def _check_correctness(self, original_method, masked_code, mask):
        original_from_masked = masked_code.replace(self._mask_symbol, mask)
        return original_method == original_from_masked