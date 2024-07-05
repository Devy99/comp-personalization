from __future__ import annotations
from utils.masker import Masker
import random, more_itertools as mit

class ConsecutiveLineMasker(Masker):
    def __init__(self, min_tokens: int = 3, max_tokens: int = None, max_lines: int = 3, **kwargs):
        super(ConsecutiveLineMasker, self).__init__(**kwargs)
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._max_lines = max_lines

    def __random_span(self, lines: list):
        random.seed(230923)
        n_lines = len(lines)
        if n_lines <= 2: return list()

        # Get all possible blocks that i can mask with at least min_tokens and at most max_tokens
        valid_blocks = list()
        for idx in range(n_lines):
            if idx + self._max_lines > n_lines: break

            blocks = self.__consecutive_blocks(lines, idx, n_lines - 1)
            for sub_block in blocks:
                start_idx, end_idx = sub_block[0], sub_block[-1]
                block_lines = lines[start_idx:end_idx + 1]

                # Tokenize each line
                tokenized_lines = list()
                for cnt, line in enumerate(block_lines):
                    tokenized_lines.extend(line.split())
                    if cnt != len(block_lines) - 1:
                        tokenized_lines.append(self._sep)

                # Skip if the number of tokens is less than the minimum
                if len(tokenized_lines) < self._min_tokens: continue
                valid_blocks.append(sub_block)

        # Remove duplicates and blocks with less than max_lines
        valid_blocks = list(set(tuple(block) for block in valid_blocks))
        valid_blocks = [list(block) for block in valid_blocks]
        
        if self._max_lines == 1:
            valid_blocks = [block for block in valid_blocks if len(block) == 1]
        else:
            valid_blocks = [block for block in valid_blocks if len(block) > 1]

        # Remove blocks that have at least one index in the blacklist
        if self._idx_blacklist:
            valid_blocks = [block for block in valid_blocks if not any(idx in self._idx_blacklist for idx in block)]

        # Select all or a random subset of blocks
        if self._max_masking > 0 and self._max_masking <= len(valid_blocks):
            valid_blocks = random.sample(valid_blocks, self._max_masking)

        valid_blocks.sort(key=lambda block: block[0])
        return valid_blocks


    def __consecutive_blocks(self, lines: list, start_idx: int, end_idx: int):
        block_indexes = list()

        # Check if there is a { before lines[start_idx]
        brackets_found = False
        temp_start_idx = start_idx - 1
        while temp_start_idx >= 0:
            if '{' in lines[temp_start_idx]: 
                brackets_found = True
                break
            temp_start_idx -= 1

        idx_before_signature = False if brackets_found else True
            
        # Don't mask the signature and the last line of the method
        if start_idx == 0 or idx_before_signature:
            # Find the first line that is next to the first { 
            while start_idx < len(lines) - 1 and not('{' in lines[start_idx]):
                start_idx += 1
            start_idx += 1
        if end_idx == len(lines) - 1:
            end_idx -= 1

        if start_idx > end_idx: return block_indexes

        idx_iter = range(start_idx, end_idx + 1)
        # If max_lines is 1, consider only single valid blocks
        if self._max_lines == 1: 
            for idx in list(idx_iter):
                block_indexes.append([idx])
            return block_indexes

        added_lines = 0
        idx_block = list()
        for idx in idx_iter:
            # Reset block if the line is empty or has only one token
            if added_lines == self._max_lines:
                block_indexes.append(idx_block)
                added_lines = 0
                idx_block = list()

            # Don't add the signature to the block if there is only one block
            if not (idx == 0 and len(lines) <= self._max_lines):
                idx_block.append(idx)

            # Skip empty lines and lines with only one token from the counter
            if not (lines[idx].strip() == '' or len(lines[idx].split()) <= 1):
                added_lines += 1

        # Add last block if it is not empty
        if idx_block: block_indexes.append(idx_block)
        return block_indexes

    def mask(self) -> list:
      random.seed(230923)
      
      # Unpack extra columns
      extra_columns_list = [None for _ in range(len(self._methods))]
      if self._extra_columns_list:
          extra_columns_list = [list(extra_cols) for extra_cols in zip(*self._extra_columns_list)]
          
      for extra_columns, method, indexes_to_mask, blacklist_indexes in zip(extra_columns_list, self._methods, self._idx_to_mask, self._idx_blacklist):
          lines = method.split(self._sep)
          n_tokens_method = len(method.split())

          if len(lines) <= 1: continue
          if self._max_masking == 0: break

          # Group contiguous added lines
          if indexes_to_mask:
            idx_blocks = [list(group) for group in mit.consecutive_groups(indexes_to_mask)]
          elif self._generate_random_if_none:
            idx_blocks = self.__random_span(lines)
          else:
             continue

          # Mask consecutive blocks
          n_added_blocks = 0
          m_extra_columns = [list() for _ in range(len(extra_columns))] if extra_columns else None
          m_masked_codes, m_masks, m_masked_indexes = list(), list(), list()
          for block_indexes in idx_blocks:
            if n_added_blocks == self._max_masking: break
            start_idx, end_idx = block_indexes[0], block_indexes[-1]

            # Divide blocks in sub-blocks of size max_lines
            sub_blocks = self.__consecutive_blocks(lines, start_idx, end_idx)
            for block in sub_blocks:
                if blacklist_indexes and any(idx in blacklist_indexes for idx in block): continue
                
                if n_added_blocks == self._max_masking: break
                block_start_idx, block_end_idx = block[0], block[-1]

                # Retrieve lines to mask
                lines_to_mask = lines[block_start_idx:block_end_idx + 1]

                # Tokenize each line
                tokenized_lines = list()
                for idx, line in enumerate(lines_to_mask):
                    tokenized_lines.extend(line.split())
                    if idx != len(lines_to_mask) - 1:
                        tokenized_lines.append(self._sep)

                # Choose a random number of tokens to mask. 
                # The masking must not exceed the 50% of the number of tokens of the method
                if self._max_tokens is None:
                    max_tokens = min(len(tokenized_lines) - 1, n_tokens_method // 2)
                else:
                    max_tokens = min(len(tokenized_lines) - 1, self._max_tokens)
                
                # Skip if the number of tokens is less than the minimum
                if max_tokens < self._min_tokens:
                    #print(f'MIN-TOKENS: Skipped block {block} for method {method} because it has less than {self._min_tokens} tokens')
                    continue
                
                n_to_mask = random.randint(self._min_tokens, max_tokens)

                # Skip if the number of tokens to mask is greater than the 50% of the number of tokens of the method
                if n_to_mask > n_tokens_method // 2:
                    #print(f'MAX-TOKENS: Skipped block {block} for method {method} because it has more than the 50% of tokens')
                    continue

                # Mask the last n_to_mask tokens
                mask = f' '.join(tokenized_lines[-n_to_mask:])
                
                # Find the masked indexes
                n_masked_lines = len(mask.split(self._sep)) + 1
                masked_indexes = list(range(block_end_idx - n_masked_lines, block_end_idx + 1))

                remaining_lines = tokenized_lines[:-n_to_mask]
                remaining_code = ' '.join(remaining_lines).strip()

                # Create the masked code with the remaining part
                remaining_code = f' {self._sep} {remaining_code}' if lines[:block_start_idx] else remaining_code
                final_sep = f' {self._sep}' if lines[block_end_idx + 1:] else ''
                masked_code = f'{self._sep}'.join(lines[:block_start_idx]).strip()   \
                           + f'{remaining_code}' \
                           + f' {self._mask_symbol}' \
                           + f'{final_sep}' \
                           + f'{self._sep}'.join(lines[block_end_idx + 1:])

                # Mask block
                if self._check_correctness(method, masked_code, mask):
                  if extra_columns:
                    for i, column in enumerate(extra_columns):
                        m_extra_columns[i].append(column)
                  m_masked_codes.append(masked_code)
                  m_masks.append(mask)
                  m_masked_indexes.extend(block)
                  n_added_blocks += 1
                else:
                    continue
                  #print("-"*50)
                  #print(f'CORRUPTED: Skipped block {block} for method {method} because it is corrupted')
                  #print(f'Original code: {method}')
                  #print(f'Masked code: {masked_code}')
                  #print(f'Mask: {mask}')
                  #print("-"*50)

          # Update list masked methods
          if extra_columns:
            for i, column in enumerate(m_extra_columns):
                self._extra_columns[i].extend(column)
          self._masked_codes.extend(m_masked_codes)
          self._masks.extend(m_masks)
          self._masked_indexes.extend(m_masked_indexes)
