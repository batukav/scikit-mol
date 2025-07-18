import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import pandas as pd
from collections import defaultdict
from typing import Union, List
from sklearn.model_selection._split import BaseShuffleSplit
from sklearn.utils.validation import _num_samples
from sklearn.utils import check_random_state


class StratifiedGroupShuffleSplit(BaseShuffleSplit):
    """Stratified ShuffleSplit cross-validator with non-overlapping groups."""

    def __init__(
        self,
        n_splits: int = 5,
        *,
        test_size: float or int = 0.2,
        train_size: float or int=None,
        random_state: int = None,
        sample_weighted: bool = False,
        suppress_warnings: bool = False
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self.sample_weighted = sample_weighted
        self.suppress_warnings = suppress_warnings

        if not suppress_warnings:
            if self.sample_weighted:
                warnings.warn(
                    f"sample_weighted = True. During the test split, groups with more samples will be prioritized",
                UserWarning,
            )

    def _iter_indices(self, X: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series], groups: Union[List, np.ndarray, pd.Series]):
        
        if y is None:
            raise ValueError(
                "StratifiedGroupShuffleSplit requires 'y' for stratification."
            )

        if groups is None:
            raise ValueError(
                "StratifiedGroupShuffleSplit requires 'groups' to be defined."
            )
        n_samples = _num_samples(X)

        if isinstance(self.test_size, float):
            n_test = int(self.test_size * n_samples)
        else:
            n_test = int(self.test_size)

        unique_groups, group_indices = np.unique(groups, return_inverse=True)
        group_counts = np.bincount(groups)
        self._check_split_viability(n_test, unique_groups, group_counts)
        n_groups = len(unique_groups)
        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = len(classes)
        overall_class_counts = np.bincount(y_indices, minlength=n_classes)

        group_info = defaultdict(
            lambda: {
                "class_counts": np.zeros(n_classes, dtype=int),
                "indices": [],
                "size": 0,
            }
        )
        for i, group_idx in enumerate(group_indices):
            class_idx = y_indices[i]
            group_info[group_idx]["class_counts"][class_idx] += 1
            group_info[group_idx]["indices"].append(i)
        for i in range(n_groups):
            group_info[i]["size"] = len(group_info[i]["indices"])

        rng = check_random_state(self.random_state)

        for _ in range(self.n_splits):
            available_groups = list(range(n_groups))
            test_groups = []

            current_test_size = 0
            current_test_counts = np.zeros(n_classes, dtype=int)

            # Phase 1: Greedily add only "safe" groups that do not exceed n_test
            while available_groups:
                safe_candidates = []
                for group_idx in available_groups:
                    group_data = group_info[group_idx]
                    if current_test_size + group_data["size"] <= n_test:
                        prospective_counts = (
                            current_test_counts + group_data["class_counts"]
                        )
                        prospective_size = current_test_size + group_data["size"]
                        ideal_counts = overall_class_counts * (
                            prospective_size / n_samples
                        )
                        error = np.sum((prospective_counts - ideal_counts) ** 2)
                        safe_candidates.append({"error": error, "id": group_idx})

                if not safe_candidates:
                    # No more groups can be added without overshooting
                    break

                safe_candidates.sort(key=lambda x: x["error"])
                pool_size = min(5, len(safe_candidates))
                candidate_pool = [cand["id"] for cand in safe_candidates[:pool_size]]
                if self.sample_weighted:
                    weights = [group_info[group_idx]["size"] for group_idx in candidate_pool]
                    best_group = rng.choice(candidate_pool, p=weights)
                else:
                    best_group = rng.choice(candidate_pool)

                test_groups.append(best_group)
                available_groups.remove(best_group)
                group_data = group_info[best_group]
                current_test_counts += group_data["class_counts"]
                current_test_size += group_data["size"]

            # Phase 2: Decide if a single overshoot is better than the current undershoot
            if available_groups and current_test_size < n_test:
                overshoot_candidates = []
                for group_idx in available_groups:
                    group_data = group_info[group_idx]
                    prospective_size = current_test_size + group_data["size"]
                    # We only care about the size difference now
                    overshoot_candidates.append(
                        {"id": group_idx, "size": prospective_size}
                    )

                if overshoot_candidates:
                    # Find the group that causes the smallest overshoot
                    overshoot_candidates.sort(key=lambda x: x["size"])
                    best_overshoot_group = overshoot_candidates[0]

                    undershoot_error = n_test - current_test_size
                    overshoot_error = best_overshoot_group["size"] - n_test

                    valid_overshoot_candidates = []
                    for cand in overshoot_candidates:
                        overshoot_error = cand["size"] - n_test
                        if overshoot_error < undershoot_error:
                            valid_overshoot_candidates.append(cand["id"])

                        if valid_overshoot_candidates:
                            # Randomly choose from the valid overshooting groups
                            best_overshoot_group_id = rng.choice(valid_overshoot_candidates)
                            test_groups.append(best_overshoot_group_id)

            test_indices = (
                np.concatenate([group_info[g_idx]["indices"] for g_idx in test_groups])
                if test_groups
                else []
            )
            if len(test_indices) == 0:
                raise RuntimeError(f"Given the dataset, no train/test split could be found. Try increasing test_size")
            all_indices = np.arange(n_samples)
            train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
            
            if isinstance(self.test_size, float):
                
                requested_test_size_ratio = self.test_size
            else:
                requested_test_size_ratio = self.test_size / n_samples
                
            test_size_error = np.abs(len(test_indices)/n_samples - requested_test_size_ratio) 
            
            if not self.suppress_warnings:
                if test_size_error > 0.05: # 5% deviation
                    warnings.warn(f"Requested and calculated test sizes differ by {test_size_error*100:.2f}%")

            yield train_indices, test_indices

    def split(self, X, y, groups=None):
        """Generates indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), optional
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : array-like of shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Each group will be kept together in either the
            train set or the test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        yield from self._iter_indices(X, y, groups)

    def get_n_splits(self):
        return self.n_splits

    def _check_split_viability(self, n_test, unique_groups, group_counts):
        too_large_groups = {}
        n_groups = 0
        for group_id, group_count in zip(unique_groups, group_counts):
            if group_count >= n_test:
                n_groups += 1
                too_large_groups[group_id] = group_count
        if len(too_large_groups) > 0 and not self.suppress_warnings and n_groups < len(unique_groups):
            warnings.warn(
                f'''
                          Some groups are too large for the test set and will never be present in the test set: {too_large_groups}.\n 
                          If you want a group to be able to be present in the test set, test_size >= group_size.
                          ''',
                UserWarning,
            )
        elif len(too_large_groups) > 0 and not self.suppress_warnings and n_groups == len(unique_groups):
            warnings.warn(
                '''
                         "Warning: All available groups are larger than the target test size. 
                         The algorithm will still try to select a group that overshoots the target, 
                         which may lead to a larger than requested test set, or an completely empty test set."
                          ''',
                UserWarning,
            )
            
