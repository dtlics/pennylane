# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Submodule for sampling a qubit mixed state.
"""
# pylint: disable=too-many-positional-arguments, too-many-arguments
from typing import Callable, Union

import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.devices.qubit.sampling import (_group_measurements,
                                              jax_random_split, sample_probs)
from pennylane.measurements import (CountsMP, ExpectationMP, SampleMeasurement,
                                    Shots)
from pennylane.measurements.classical_shadow import (ClassicalShadowMP,
                                                     ShadowExpvalMP)
from pennylane.ops import LinearCombination, Sum
from pennylane.typing import TensorLike

from .apply_operation import _get_num_wires, apply_operation
from .measure import measure


def _apply_diagonalizing_gates(
    mps: list[SampleMeasurement], state: np.ndarray, is_state_batched: bool = False
):
    if len(mps) == 1:
        diagonalizing_gates = mps[0].diagonalizing_gates()
    elif all(mp.obs for mp in mps):
        diagonalizing_gates = qml.pauli.diagonalize_qwc_pauli_words([mp.obs for mp in mps])[0]
    else:
        diagonalizing_gates = []

    for op in diagonalizing_gates:
        state = apply_operation(op, state, is_state_batched=is_state_batched)

    return state


def _process_samples(
    mp,
    samples,
    wire_order,
):
    """Processes samples like SampleMP.process_samples, but different in need of some special cases e.g. CountsMP"""
    wire_map = dict(zip(wire_order, range(len(wire_order))))
    mapped_wires = [wire_map[w] for w in mp.wires]

    if mapped_wires:
        # if wires are provided, then we only return samples from those wires
        samples = samples[..., mapped_wires]

    num_wires = samples.shape[-1]  # wires is the last dimension

    if mp.obs is None:
        # if no observable was provided then return the raw samples
        return samples

    # Replace the basis state in the computational basis with the correct eigenvalue.
    # Extract only the columns of the basis samples required based on ``wires``.
    # This step converts e.g. 110 -> 6
    powers_of_two = 2 ** qml.math.arange(num_wires)[::-1]  # e.g. [1, 2, 4, ...]
    indices = qml.math.array(samples @ powers_of_two)
    return mp.eigvals()[indices]


def _process_expval_samples(processed_sample):
    """Processes a set of samples and returns the expectation value of an observable."""
    eigvals, counts = math.unique(processed_sample, return_counts=True)
    probs = counts / math.sum(counts)
    return math.dot(probs, eigvals)


def _process_variance_samples(processed_sample):
    """Processes a set of samples and returns the variance of an observable."""
    eigvals, counts = math.unique(processed_sample, return_counts=True)
    probs = counts / math.sum(counts)
    return math.dot(probs, (eigvals**2)) - math.dot(probs, eigvals) ** 2


# pylint:disable = too-many-arguments
def _measure_with_samples_diagonalizing_gates(
    mps: list[SampleMeasurement],
    state: np.ndarray,
    shots: Shots,
    is_state_batched: bool = False,
    rng=None,
    prng_key=None,
    readout_errors: list[Callable] = None,
) -> TensorLike:
    """Returns the samples of the measurement process performed on the given state,
    by rotating the state into the measurement basis using the diagonalizing gates
    given by the measurement process.

    Args:
        mp (~.measurements.SampleMeasurement): The sample measurement to perform
        state (TensorLike): The state vector to sample from
        shots (~.measurements.Shots): The number of samples to take
        is_state_batched (bool): whether the state is batched or not
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
        readout_errors (List[Callable]): List of channels to apply to each wire being measured
        to simulate readout errors.

    Returns:
        TensorLike[Any]: Sample measurement results
    """
    # apply diagonalizing gates
    state = _apply_diagonalizing_gates(mps, state, is_state_batched)

    total_indices = _get_num_wires(state, is_state_batched)
    wires = qml.wires.Wires(range(total_indices))

    def _process_single_shot(samples):
        processed = []
        for mp in mps:
            res = mp.process_samples(samples, wires)
            if not isinstance(mp, CountsMP):
                res = qml.math.squeeze(res)

            processed.append(res)

        return tuple(processed)

    try:
        prng_key, _ = jax_random_split(prng_key)
        samples = sample_state(
            state,
            shots=shots.total_shots,
            is_state_batched=is_state_batched,
            wires=wires,
            rng=rng,
            prng_key=prng_key,
            readout_errors=readout_errors,
        )
    except ValueError as e:
        if str(e) != "probabilities contain NaN":
            raise e
        samples = qml.math.full((shots.total_shots, len(wires)), 0)

    processed_samples = []
    for lower, upper in shots.bins():
        shot = _process_single_shot(samples[..., lower:upper, :])
        processed_samples.append(shot)

    if shots.has_partitioned_shots:
        return tuple(zip(*processed_samples))

    return processed_samples[0]


def _measure_classical_shadow(
    mp: list[Union[ClassicalShadowMP, ShadowExpvalMP]],
    state: np.ndarray,
    shots: Shots,
    is_state_batched: bool = False,
    rng=None,
    prng_key=None,
    readout_errors=None,
):
    """
    Returns the result of a classical shadow measurement on the given state.

    A classical shadow measurement doesn't fit neatly into the current measurement API
    since different diagonalizing gates are used for each shot. Here it's treated as a
    state measurement with shots instead of a sample measurement.

    Args:
        mp (~.measurements.SampleMeasurement): The sample measurement to perform
        state (np.ndarray[complex]): The state vector to sample from
        shots (~.measurements.Shots): The number of samples to take
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.

    Returns:
        TensorLike[Any]: Sample measurement results
    """
    # pylint: disable=unused-argument

    # the list contains only one element based on how we group measurements
    mp = mp[0]

    wires = qml.wires.Wires(range(len(state.shape)))

    if shots.has_partitioned_shots:
        return [tuple(mp.process_state_with_shots(state, wires, s, rng=rng) for s in shots)]

    return [mp.process_state_with_shots(state, wires, shots.total_shots, rng=rng)]


def _measure_hamiltonian_with_samples(
    mp: list[SampleMeasurement],
    state: np.ndarray,
    shots: Shots,
    is_state_batched: bool = False,
    rng=None,
    prng_key=None,
    readout_errors=None,
):
    # the list contains only one element based on how we group measurements
    mp = mp[0]

    # if the measurement process involves a Hamiltonian, measure each
    # of the terms separately and sum
    def _sum_for_single_shot(s, prng_key=None):
        results = measure_with_samples(
            [ExpectationMP(t) for t in mp.obs.terms()[1]],
            state,
            s,
            is_state_batched=is_state_batched,
            rng=rng,
            prng_key=prng_key,
            readout_errors=readout_errors,
        )
        return sum(c * res for c, res in zip(mp.obs.terms()[0], results))

    keys = jax_random_split(prng_key, num=shots.num_copies)
    unsqueezed_results = tuple(
        _sum_for_single_shot(type(shots)(s), key) for s, key in zip(shots, keys)
    )
    return [unsqueezed_results] if shots.has_partitioned_shots else [unsqueezed_results[0]]


def _measure_sum_with_samples(
    mp: list[SampleMeasurement],
    state: np.ndarray,
    shots: Shots,
    is_state_batched: bool = False,
    rng=None,
    prng_key=None,
    readout_errors: list[Callable] = None,
):
    """Compute expectation values of Sum Observables"""
    mp = mp[0]

    def _sum_for_single_shot(s, prng_key=None):
        results = measure_with_samples(
            [ExpectationMP(t) for t in mp.obs],
            state,
            s,
            is_state_batched=is_state_batched,
            rng=rng,
            prng_key=prng_key,
            readout_errors=readout_errors,
        )
        return sum(results)

    keys = jax_random_split(prng_key, num=shots.num_copies)
    unsqueezed_results = tuple(
        _sum_for_single_shot(type(shots)(s), key) for s, key in zip(shots, keys)
    )
    return [unsqueezed_results] if shots.has_partitioned_shots else [unsqueezed_results[0]]


def sample_state(
    state,
    shots: int,
    is_state_batched: bool = False,
    wires=None,
    rng=None,
    prng_key=None,
    readout_errors: list[Callable] = None,
) -> np.ndarray:
    """Returns a series of computational basis samples of a state.

    Args:
        state (array[complex]): A state vector to be sampled
        shots (int): The number of samples to take
        is_state_batched (bool): whether the state is batched or not
        wires (Sequence[int]): The wires to sample
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]):
            A seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
        readout_errors (List[Callable]): List of channels to apply to each wire being measured
        to simulate readout errors.

    Returns:
        ndarray[int]: Sample values of the shape (shots, num_wires)
    """

    total_indices = _get_num_wires(state, is_state_batched)
    state_wires = qml.wires.Wires(range(total_indices))

    wires_to_sample = wires or state_wires
    num_wires = len(wires_to_sample)

    with qml.queuing.QueuingManager.stop_recording():
        probs = measure(qml.probs(wires=wires_to_sample), state, is_state_batched, readout_errors)

    # After getting the correct probs, there's no difference between mixed states and pure states.
    # Therefore, we directly re-use the sample_probs from the module qubit.
    return sample_probs(probs, shots, num_wires, is_state_batched, rng, prng_key=prng_key)


def measure_with_samples(
    measurements: list[Union[SampleMeasurement, ClassicalShadowMP, ShadowExpvalMP]],
    state: np.ndarray,
    shots: Shots,
    is_state_batched: bool = False,
    rng=None,
    prng_key=None,
    readout_errors: list[Callable] = None,
) -> TensorLike:
    """Returns the samples of the measurement process performed on the given state.
    This function assumes that the user-defined wire labels in the measurement process
    have already been mapped to integer wires used in the device.

    Args:
        mp (SampleMeasurement): The sample measurement to perform
        state (np.ndarray[complex]): The state vector to sample from
        shots (Shots): The number of samples to take
        is_state_batched (bool): whether the state is batched or not
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
        readout_errors (List[Callable]): List of channels to apply to each wire being measured
        to simulate readout errors.

    Returns:
        TensorLike[Any]: Sample measurement results
    """
    groups, indices = _group_measurements(measurements)
    all_res = []
    for group in groups:
        if isinstance(group[0], ExpectationMP) and isinstance(group[0].obs, LinearCombination):
            measure_fn = _measure_hamiltonian_with_samples
        elif isinstance(group[0], ExpectationMP) and isinstance(group[0].obs, Sum):
            measure_fn = _measure_sum_with_samples
        elif isinstance(group[0], (ClassicalShadowMP, ShadowExpvalMP)):
            measure_fn = _measure_classical_shadow
        else:
            # measure with the usual method (rotate into the measurement basis)
            measure_fn = _measure_with_samples_diagonalizing_gates

        prng_key, key = jax_random_split(prng_key)
        all_res.extend(
            measure_fn(
                group,
                state,
                shots,
                is_state_batched=is_state_batched,
                rng=rng,
                prng_key=key,
                readout_errors=readout_errors,
            )
        )

    flat_indices = [_i for i in indices for _i in i]

    # reorder results
    sorted_res = tuple(
        res for _, res in sorted(list(enumerate(all_res)), key=lambda r: flat_indices[r[0]])
    )

    # put the shot vector axis before the measurement axis
    if shots.has_partitioned_shots:
        sorted_res = tuple(zip(*sorted_res))

    return sorted_res
