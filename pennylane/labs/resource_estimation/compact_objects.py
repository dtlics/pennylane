# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the base classes for compact hamiltonians and states
to be used with the existing resource estimation pipeline"""
from pennylane.labs.resource_estimation import (
    ResourceAdjoint,
    ResourceBasisRotation,
    ResourceControlled,
    ResourceMultiRZ,
    ResourcePauliRot,
    ResourceX,
    ResourceY,
    ResourceZ,
)


class CompactLCU:
    """A class storing the meta data associated with an LCU decomposition of an operator."""

    def __init__(
        self,
        num_wires,
        lcu_type=None,
        num_terms=None,
        k_local=None,
        cost_per_term=None,
        cost_per_exp_term=None,
        cost_per_ctrl_exp_term=None,
        one_norm_error=None,
    ) -> None:
        """Store the meta info into the class attributes."""
        self.num_wires = num_wires
        self.lcu_type = lcu_type
        self.num_terms = num_terms
        self.k_local = k_local
        self.cost_per_term = cost_per_term
        self.cost_per_exp_term = cost_per_exp_term
        self.cost_per_ctrl_exp_term = cost_per_ctrl_exp_term
        self.one_norm_error = one_norm_error

    def info(self, print_info=False):
        """Return a dictionary of the metadata or display it on screen."""
        metadata_dict = self.__dict__

        if print_info:
            print(f"CompactLCU(num_wires={metadata_dict["num_wires"]}):")
            for k, v in metadata_dict.items():
                if k == "num_wires":
                    continue
                print(f"-> {k}: {v}")

        return metadata_dict

    def update(self):
        """Update empty information after initializing the class."""
        if self.lcu_type == "pauli":
            cost_per_term = {}
            x, y, z = (ResourceX.resource_rep(), ResourceY.resource_rep(), ResourceZ.resource_rep())

            freq = self.k_local // 3

            cost_per_term[x] = freq
            cost_per_term[z] = freq
            cost_per_term[y] = self.k_local - freq

            avg_pword = freq * "X" + freq * "Z" + (self.k_local - freq) * "Y"
            cost_per_exp_term = {ResourcePauliRot.resource_rep(avg_pword): 1}

            self.cost_per_term = cost_per_term
            self.cost_per_exp_term = cost_per_exp_term

        if self.lcu_type == "cdf":
            cost_per_term = {}
            cost_per_exp_term = {}
            cost_per_ctrl_exp_term = {}

            basis_rot = ResourceBasisRotation.resource_rep(2**self.num_wires)
            adj_basis_rot = ResourceAdjoint.resource_rep(
                ResourceBasisRotation, {"dim_N": 2**self.num_wires}
            )
            z = ResourceZ.resource_rep()
            multi_z = ResourceMultiRZ.resource_rep(2)
            ctrl_multi_z = ResourceControlled.resource_rep(
                ResourceMultiRZ, {"num_wires": 2}, 1, 0, 0
            )

            cost_per_term[basis_rot] = 2
            cost_per_term[adj_basis_rot] = 2
            cost_per_term[z] = 2

            cost_per_exp_term[basis_rot] = 2
            cost_per_exp_term[adj_basis_rot] = 2
            cost_per_exp_term[multi_z] = 1

            cost_per_ctrl_exp_term[basis_rot] = 2
            cost_per_ctrl_exp_term[adj_basis_rot] = 2
            cost_per_ctrl_exp_term[ctrl_multi_z] = 1

            self.cost_per_term = cost_per_term
            self.cost_per_exp_term = cost_per_exp_term
            self.cost_per_ctrl_exp_term = cost_per_ctrl_exp_term

        return


class CompactState:
    """A class storing the meta data associated with a quantum state."""

    def __init__(
        self,
        num_wires,
        data_size=None,
        is_sparse=False,
        is_bitstring=False,
        precision=None,
        num_aux_wires=None,
        cost_per_prep=None,
        cost_per_ctrl_prep=None,
    ) -> None:
        self.num_wires = num_wires
        self.data_size = data_size
        self.is_sparse = is_sparse
        self.is_bitstring = is_bitstring
        self.precision = precision
        self.num_aux_wires = num_aux_wires
        self.cost_per_prep = cost_per_prep
        self.cost_per_ctrl_prep = cost_per_ctrl_prep

    def info(self, print_info=False):
        """Return a dictionary of the metadata or display it on screen."""
        metadata_dict = self.__dict__

        if print_info:
            print(f"CompactState(num_wires={metadata_dict["num_wires"]}):")
            for k, v in metadata_dict.items():
                if k == "num_wires":
                    continue
                print(f"-> {k}: {v}")

        return metadata_dict

    def update(self):
        """Update empty information after initializing the class."""
        return
