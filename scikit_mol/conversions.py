from collections.abc import Sequence
from typing import Optional, Union
from types import ModuleType
import rdkit

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.rdBase import BlockLogs
from sklearn.base import BaseEstimator, TransformerMixin
from rdkit.Chem.Scaffolds import MurckoScaffold

from scikit_mol._constants import DOCS_BASE_URL
from scikit_mol.core import (
    InvalidMol,
    NoFitNeededMixin,
    check_transform_input,
    feature_names_default_mol,
)
from scikit_mol.parallel import parallelized_with_batches

# from scikit_mol._invalid import InvalidMol


class SmilesToMolTransformer(TransformerMixin, NoFitNeededMixin, BaseEstimator):
    """
    Transformer for converting SMILES strings to RDKit mol objects.

    This transformer can be included in pipelines during development and training,
    but the safe inference mode should only be enabled when deploying models for
    inference in production environments.
    """

    _doc_link_module = "scikit_mol"
    _doc_link_template = (
        DOCS_BASE_URL + "{estimator_module}/#{estimator_module}.{estimator_name}"
    )

    def __init__(
        self, n_jobs: Optional[None] = None, safe_inference_mode: bool = False
    ):
        """
        Parameters
        -----------
        n_jobs : int, optional default=None
            The maximum number of concurrently running jobs.
            `None` is a marker for 'unset' that will be interpreted as `n_jobs=1` unless the call is performed under a `parallel_config()` context manager that sets another value for `n_jobs`.
        safe_inference_mode : bool, default=False
            If `True`, enables safeguards for handling invalid data during inference.
            This should only be set to `True` when deploying models to production.
        """
        self.n_jobs = n_jobs
        self.safe_inference_mode = safe_inference_mode

    @feature_names_default_mol
    def get_feature_names_out(self, input_features=None):
        return input_features

    def fit(self, X=None, y=None):
        """Included for scikit-learn compatibility, does nothing"""
        return self

    def transform(
        self, X_smiles_list: Sequence[str], y=None
    ) -> NDArray[Union[Chem.Mol, InvalidMol]]:
        """Converts SMILES into RDKit mols

        Parameters
        ----------
        X_smiles_list : Sequence[str]
            sequence of SMILES strings to transform

        Returns
        -------
        NDArray[Union[Chem.Mol, InvalidMol]]
            Array of RDKit mol objects or InvalidMol objects if a SMILES string is invalid and `safe_inference_mode=True`
        Raises
        ------
        ValueError
            Raises ValueError if a SMILES string is unparsable by RDKit and `safe_inference_mode=False`
        """

        arrays = parallelized_with_batches(self._transform, X_smiles_list, self.n_jobs)
        arr = np.concatenate(arrays)
        return arr

    @check_transform_input
    def _transform(self, X):
        X_out = []
        with BlockLogs():
            for smiles in X:
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
                if mol:
                    errors = Chem.DetectChemistryProblems(mol)
                    if errors:
                        error_message = "\n".join(error.Message() for error in errors)
                        message = f"Invalid Molecule: {error_message}"
                        X_out.append(InvalidMol(str(self), message))
                    else:
                        Chem.SanitizeMol(mol)
                        X_out.append(mol)
                else:
                    message = f"Invalid SMILES: {smiles}"
                    X_out.append(InvalidMol(str(self), message))
        if not self.safe_inference_mode and not all(X_out):
            fails = [x for x in X_out if not x]
            raise ValueError(
                f"Invalid input found: {fails}."
            )  # TODO with this approach we get all errors, but we do process ALL the smiles first which could be slow
        return np.array(X_out).reshape(-1, 1)

    @check_transform_input
    def inverse_transform(self, X_mols_list, y=None):
        X_out = []

        for mol in X_mols_list:
            if isinstance(mol, Chem.Mol):
                try:
                    smiles = Chem.MolToSmiles(mol)
                    X_out.append(smiles)
                except Exception as e:
                    X_out.append(
                        InvalidMol(str(self), f"Error converting Mol to SMILES: {e}")
                    )
            else:
                X_out.append(InvalidMol(str(self), f"Not a Mol: {mol}"))

        if not self.safe_inference_mode and not all(isinstance(x, str) for x in X_out):
            fails = [x for x in X_out if not isinstance(x, str)]
            raise ValueError(f"Invalid Mols found: {fails}.")

        return np.array(X_out).reshape(-1, 1)


class MolToScaffoldTransformer(SmilesToMolTransformer):
    """
    Transformer for converting SMILES strings to molecular scaffolds.
    First converts SMILES to RDKit mol objects, then extracts the scaffold.
    """

    def __init__(
        self,
        n_jobs: Optional[None] = None,
        safe_inference_mode: bool = False,
        scaffold_transformer: ModuleType = MurckoScaffold,
    ):
        super().__init__(n_jobs, safe_inference_mode)
        self.scaffold_transformer = scaffold_transformer

        # Question: How generic should be the scaffold transformer?
        # For now, just for demonstration I'm using the MurckoScaffold class

    def transform(
        self, X_smiles_list: Sequence[str]
    ) -> NDArray[Union[Chem.Mol, InvalidMol]]:
        # First step: convert SMILES to molecules using parent class
        mols = super().transform(X_smiles_list, y=None).flatten()

        self.mols = mols  # to be deleted
        # Second step: convert molecules to scaffolds
        scaffolds = (
            []
        )  # TODO: this will be very slow for large datasets, improve efficiency via initializing list
        for mol in mols:
            if isinstance(mol, Chem.Mol):
                try:
                    scaffold = self.scaffold_transformer.GetScaffoldForMol(mol)
                    scaffolds.append(scaffold)
                except Exception as e:
                    scaffolds.append(
                        InvalidMol(str(self), f"Error creating scaffold: {e}")
                    )
            else:
                scaffolds.append(mol)  # Keep InvalidMol objects as is

        self.scaffolds = np.array(scaffolds).reshape(-1, 1).flatten()
        return np.array(scaffolds).reshape(-1, 1)

    def get_unique_scaffold_ids(self) -> NDArray[np.int_]:

        scaffold_smiles = self._create_smiles_from_mol()
        # Get unique labels
        _, labels = np.unique(scaffold_smiles, return_inverse=True)

        return labels

    def _create_smiles_from_mol(self):

        if type(self.scaffolds) == rdkit.Chem.rdchem.Mol:
            return Chem.MolToSmiles(self.scaffolds)
        elif type(self.scaffolds) == np.ndarray:
            scaffold_smiles = []
            for scaffold in self.scaffolds:
                scaffold_smiles.append(Chem.MolToSmiles(scaffold))

            return scaffold_smiles
        else:
            raise RuntimeError("Unknown data type ")
        # Keep scaffold_smiles ??
        # self.scaffold_smiles = scaffold_smiles
