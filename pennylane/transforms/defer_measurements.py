# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Code for the tape transform implementing the deferred measurement principle."""

from functools import lru_cache, partial
from numbers import Number
from typing import Optional, Sequence, Union
from warnings import warn

import pennylane as qml
from pennylane.measurements import (
    CountsMP,
    MeasurementValue,
    MidMeasureMP,
    ProbabilityMP,
    SampleMP,
    get_mcm_predicates,
)
from pennylane.ops.op_math import ctrl
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn
from pennylane.wires import Wires

# pylint: disable=too-many-branches, protected-access, too-many-statements


def _check_tape_validity(tape: QuantumScript):
    """Helper function to check that the tape is valid."""
    cv_types = (qml.operation.CVOperation, qml.operation.CVObservable)
    ops_cv = any(isinstance(op, cv_types) and op.name != "Identity" for op in tape.operations)
    obs_cv = any(
        isinstance(getattr(op, "obs", None), cv_types)
        and not isinstance(getattr(op, "obs", None), qml.Identity)
        for op in tape.measurements
    )
    if ops_cv or obs_cv:
        raise ValueError("Continuous variable operations and observables are not supported.")

    for mp in tape.measurements:
        if isinstance(mp, (CountsMP, ProbabilityMP, SampleMP)) and not (
            mp.obs or mp._wires or mp.mv is not None
        ):
            raise ValueError(
                f"Cannot use {mp.__class__.__name__} as a measurement without specifying wires "
                "when using qml.defer_measurements. Deferred measurements can occur "
                "automatically when using mid-circuit measurements on a device that does not "
                "support them."
            )

        if mp.__class__.__name__ == "StateMP":
            raise ValueError(
                "Cannot use StateMP as a measurement when using qml.defer_measurements. "
                "Deferred measurements can occur automatically when using mid-circuit "
                "measurements on a device that does not support them."
            )

    samples_present = any(isinstance(mp, SampleMP) for mp in tape.measurements)
    postselect_present = any(
        op.postselect is not None for op in tape.operations if isinstance(op, MidMeasureMP)
    )
    if postselect_present and samples_present and tape.batch_size is not None:
        raise ValueError(
            "Returning qml.sample is not supported when postselecting mid-circuit "
            "measurements with broadcasting"
        )


def _collect_mid_measure_info(tape: QuantumScript):
    """Helper function to collect information related to mid-circuit measurements in the tape."""

    # Find wires that are reused after measurement
    measured_wires = []
    reused_measurement_wires = set()
    any_repeated_measurements = False
    is_postselecting = False

    for op in tape:
        if isinstance(op, MidMeasureMP):
            if op.postselect is not None:
                is_postselecting = True
            if op.reset:
                reused_measurement_wires.add(op.wires[0])

            if op.wires[0] in measured_wires:
                any_repeated_measurements = True
            measured_wires.append(op.wires[0])

        else:
            reused_measurement_wires = reused_measurement_wires.union(
                set(measured_wires).intersection(op.wires.toset())
            )

    return measured_wires, reused_measurement_wires, any_repeated_measurements, is_postselecting


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


@lru_cache
def _get_plxpr_defer_measurements():
    try:
        # pylint: disable=import-outside-toplevel
        import jax

        from pennylane.capture import PlxprInterpreter
        from pennylane.capture.primitives import cond_prim, ctrl_transform_prim, measure_prim
    except ImportError:  # pragma: no cover
        return None, None

    # pylint: disable=redefined-outer-name

    class DeferMeasurementsInterpreter(PlxprInterpreter):
        """Interpreter for applying the defer_measurements transform to plxpr."""

        # pylint: disable=unnecessary-lambda-assignment,attribute-defined-outside-init,no-self-use

        def __init__(self, aux_wires):
            super().__init__()
            self._aux_wires = Wires(aux_wires)

            # We use a dict here instead of a normal int variable because we want the state to mutate
            # when we interpret higher-order primitives
            self.state = {"cur_idx": 0}

        def cleanup(self) -> None:
            """Perform any final steps after iterating through all equations.

            Blank by default, this method can clean up instance variables. Particularly,
            this method can be used to deallocate qubits and registers when converting to
            a Catalyst variant jaxpr.
            """
            self.state = {"cur_idx": 0}

        def interpret_dynamic_operation(self, data, struct, idx):
            """Interpret an operation that uses mid-circuit measurement outcomes as parameters.

            * This will not work if mid-circuit measurement values are used to specify
              operator wires.
            * This will not work if more than one parameter uses mid-circuit measurement values.

            Args:
                data (TensorLike): Flattened data of the operator
                struct (PyTreeDef): Pytree structure of the operator
                idx (int): Index of mid-circuit measurement value in ``data``

            Returns:
                None
            """
            mv = data[idx]
            for branch, value in mv.items():
                data[idx] = value
                op = jax.tree_util.tree_unflatten(struct, data)
                qml.ctrl(op, mv.wires, control_values=branch)

        def interpret_operation(self, op: "pennylane.operation.Operator"):
            """Interpret a PennyLane operation instance.

            Args:
                op (Operator): a pennylane operator instance

            Returns:
                Any

            This method is only called when the operator's output is a dropped variable,
            so the output will not affect later equations in the circuit.

            See also: :meth:`~.interpret_operation_eqn`.

            """
            data, struct = jax.tree_util.tree_flatten(op)

            idx = -1
            for i, d in enumerate(data):
                if isinstance(d, MeasurementValue):
                    idx = i
                    break
            if idx != -1:
                return self.interpret_dynamic_operation(data, struct, idx)

            return jax.tree_util.tree_unflatten(struct, data)

        def interpret_measurement(self, measurement: "qml.measurement.MeasurementProcess"):
            """Interpret a measurement process instance.

            Args:
                measurement (MeasurementProcess): a measurement instance.

            See also :meth:`~.interpret_measurement_eqn`.

            """
            if measurement.mv is not None:
                kwargs = {"wires": measurement.wires, "eigvals": measurement.eigvals()}
                if isinstance(measurement, CountsMP):
                    kwargs["all_outcomes"] = measurement.all_outcomes

                measurement = type(measurement)(**kwargs)

            return super().interpret_measurement(measurement)

        def resolve_mcm_values(
            self, eqn: "jax.core.JaxprEqn", invals: Sequence[Union[MeasurementValue, Number]]
        ) -> MeasurementValue:
            """Create a ``MeasurementValue`` that captures all classical processing of the
            input ``eqn`` in its ``processing_fn``.

            Args:
                eqn (jax.core.JaxprEqn): Jaxpr equation containing the primitive to apply
                invals (Sequence[Union[MeasurementValue, Number]]): Inputs to the equation

            Returns:
                MeasurementValue: ``MeasurementValue`` containing classical processing information
                for applying the input equation to mid-circuit measurement outcomes.
            """
            # pylint: disable=protected-access,unnecessary-lambda
            assert len(invals) <= 2

            if len(invals) == 1:
                meas_val = invals[0]
                processing_fn = lambda x: eqn.primitive.bind(x, **eqn.params)
                return meas_val._apply(processing_fn)

            if all(isinstance(inval, MeasurementValue) for inval in invals):
                processing_fn = lambda *x: eqn.primitive.bind(*x, **eqn.params)
                return invals[0]._transform_bin_op(processing_fn, invals[1])

            # One MeasurementValue, one number
            [meas_val, other] = invals if isinstance(invals[0], MeasurementValue) else invals[::-1]
            processing_fn = lambda x: eqn.primitive.bind(x, other, **eqn.params)
            return meas_val._apply(processing_fn)

        def eval(self, jaxpr: "jax.core.Jaxpr", consts: list, *args) -> list:
            """Evaluate a jaxpr.

            Args:
                jaxpr (jax.core.Jaxpr): the jaxpr to evaluate
                consts (list[TensorLike]): the constant variables for the jaxpr
                *args (tuple[TensorLike]): The arguments for the jaxpr.

            Returns:
                list[TensorLike]: the results of the execution.

            """
            self._env = {}
            self.setup()

            for arg, invar in zip(args, jaxpr.invars, strict=True):
                self._env[invar] = arg
            for const, constvar in zip(consts, jaxpr.constvars, strict=True):
                self._env[constvar] = const

            for eqn in jaxpr.eqns:
                primitive = eqn.primitive
                custom_handler = self._primitive_registrations.get(primitive, None)

                if custom_handler:
                    invals = [self.read(invar) for invar in eqn.invars]
                    outvals = custom_handler(self, *invals, **eqn.params)
                elif getattr(primitive, "prim_type", "") == "operator":
                    outvals = self.interpret_operation_eqn(eqn)
                elif getattr(primitive, "prim_type", "") == "measurement":
                    outvals = self.interpret_measurement_eqn(eqn)
                else:
                    invals = [self.read(invar) for invar in eqn.invars]
                    if any(isinstance(inval, MeasurementValue) for inval in invals):
                        outvals = self.resolve_mcm_values(eqn, invals)
                    else:
                        outvals = primitive.bind(*invals, **eqn.params)

                if not primitive.multiple_results:
                    outvals = [outvals]
                for outvar, outval in zip(eqn.outvars, outvals, strict=True):
                    self._env[outvar] = outval

            # Read the final result of the Jaxpr from the environment
            outvals = []
            for var in jaxpr.outvars:
                outval = self.read(var)
                if isinstance(outval, qml.operation.Operator):
                    outvals.append(self.interpret_operation(outval))
                else:
                    outvals.append(outval)
            self.cleanup()
            self._env = {}
            return outvals

    @DeferMeasurementsInterpreter.register_primitive(measure_prim)
    def _(self, wires, reset=False, postselect=None):
        if self.state["cur_idx"] >= len(self._aux_wires):
            raise ValueError(
                "Not enough auxiliary wires provided to apply specified number of mid-circuit "
                "measurements using qml.defer_measurements."
            )

        meas = type.__call__(
            MidMeasureMP,
            Wires(self._aux_wires[self.state["cur_idx"]]),
            reset=reset,
            postselect=postselect,
            id=self.state["cur_idx"],
        )

        cnot_wires = (wires, self._aux_wires[self.state["cur_idx"]])
        if postselect is not None:
            qml.Projector(jax.numpy.array([postselect]), wires=wires)

        qml.CNOT(wires=cnot_wires)
        if reset:
            if postselect is None:
                qml.CNOT(wires=cnot_wires[::-1])
            elif postselect == 1:
                qml.PauliX(wires=wires)

        self.state["cur_idx"] += 1
        return MeasurementValue([meas], lambda x: x)

    @DeferMeasurementsInterpreter.register_primitive(cond_prim)
    def _(
        self, *invals, jaxpr_branches, consts_slices, args_slice
    ):  # pylint: disable=unused-argument
        n_branches = len(jaxpr_branches)
        conditions = invals[:n_branches]
        if not any(isinstance(c, MeasurementValue) for c in conditions):
            return PlxprInterpreter._primitive_registrations[cond_prim](
                self,
                *invals,
                jaxpr_branches=jaxpr_branches,
                consts_slices=consts_slices,
                args_slice=args_slice,
            )

        conditions = get_mcm_predicates(conditions[:-1])
        args = invals[args_slice]

        for i, (condition, jaxpr) in enumerate(zip(conditions, jaxpr_branches, strict=True)):
            if jaxpr is None:
                continue

            if isinstance(condition, MeasurementValue):
                control_wires = Wires([m.wires[0] for m in condition.measurements])

                for branch, value in condition.items():
                    cur_consts = invals[consts_slices[i]]
                    qml.cond(value, ctrl_transform_prim.bind)(
                        *cur_consts,
                        *args,
                        *control_wires,
                        jaxpr=jaxpr,
                        n_control=len(control_wires),
                        control_values=branch,
                        work_wires=None,
                        n_consts=len(cur_consts),
                    )

        return []

    def defer_measurements_plxpr_to_plxpr(
        jaxpr, consts, targs, tkwargs, *args
    ):  # pylint: disable=unused-argument

        if (aux_wires := tkwargs.pop("aux_wires", None)) is None:
            raise ValueError(
                "'aux_wires' argument for qml.defer_measurements must be provided "
                "when qml.capture.enabled() is True."
            )
        if tkwargs.pop("reduce_postselected", False):
            warn(
                "Cannot set 'reduce_postselected=True' with qml.capture.enabled() "
                "when using qml.defer_measurements. Argument will be ignored.",
                UserWarning,
            )

        interpreter = DeferMeasurementsInterpreter(aux_wires=aux_wires)

        def wrapper(*inner_args):
            return interpreter.eval(jaxpr, consts, *inner_args)

        return jax.make_jaxpr(wrapper)(*args)

    return DeferMeasurementsInterpreter, defer_measurements_plxpr_to_plxpr


DeferMeasurementsInterpreter, defer_measurements_plxpr_to_plxpr = _get_plxpr_defer_measurements()


# pylint: disable=unused-argument
@partial(transform, plxpr_transform=defer_measurements_plxpr_to_plxpr)
def defer_measurements(
    tape: QuantumScript,
    reduce_postselected: bool = True,
    allow_postselect: bool = True,
    aux_wires: Optional[Union[int, Sequence[int], Wires]] = None,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Quantum function transform that substitutes operations conditioned on
    measurement outcomes to controlled operations.

    This transform uses the `deferred measurement principle
    <https://en.wikipedia.org/wiki/Deferred_Measurement_Principle>`_ and
    applies to qubit-based quantum functions.

    Support for mid-circuit measurements is device-dependent. If a device
    doesn't support mid-circuit measurements natively, then the QNode will
    apply this transform.

    .. note::

        The transform uses the :func:`~.ctrl` transform to implement operations
        controlled on mid-circuit measurement outcomes. The set of operations
        that can be controlled as such depends on the set of operations
        supported by the chosen device.

    .. note::

        Devices that inherit from :class:`~pennylane.devices.QubitDevice` **must** be initialized
        with an additional wire for each mid-circuit measurement after which the measured
        wire is reused or reset for ``defer_measurements`` to transform the quantum tape
        correctly.

    .. note::

        This transform does not change the list of terminal measurements returned by
        the quantum function.

    .. note::

        When applying the transform on a quantum function that contains the
        :class:`~.Snapshot` instruction, state information corresponding to
        simulating the transformed circuit will be obtained. No
        post-measurement states are considered.

    .. warning::

        :func:`~.pennylane.state` is not supported with the ``defer_measurements`` transform.
        Additionally, :func:`~.pennylane.probs`, :func:`~.pennylane.sample` and
        :func:`~.pennylane.counts` can only be used with ``defer_measurements`` if wires
        or an observable are explicitly specified.

    .. warning::

        ``defer_measurements`` does not support using custom wire labels if any measured
        wires are reused or reset.

    Args:
        tape (QNode or QuantumTape or Callable): a quantum circuit.
        reduce_postselected (bool): Whether to use postselection information to reduce the number
            of operations and control wires in the output tape. Active by default. This is currently
            ignored if program capture is enabled.
        allow_postselect (bool): Whether postselection is allowed. In order to perform postselection
            with ``defer_measurements``, the device must support the :class:`~.Projector` operation.
            Defaults to ``True``. This is currently ignored if program capture is enabled.
        aux_wires (Sequence): Optional sequence of wires to use to map mid-circuit measurements. This is
            only used if program capture is enabled.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The
            transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    Raises:
        ValueError: If custom wire labels are used with qubit reuse or reset
        ValueError: If any measurements with no wires or observable are present
        ValueError: If continuous variable operations or measurements are present
        ValueError: If using the transform with any device other than
            :class:`default.qubit <~pennylane.devices.DefaultQubit>` and postselection is used

    **Example**

    Suppose we have a quantum function with mid-circuit measurements and
    conditional operations:

    .. code-block:: python3

        def qfunc(par):
            qml.RY(0.123, wires=0)
            qml.Hadamard(wires=1)
            m_0 = qml.measure(1)
            qml.cond(m_0, qml.RY)(par, wires=0)
            return qml.expval(qml.Z(0))

    The ``defer_measurements`` transform allows executing such quantum
    functions without having to perform mid-circuit measurements:

    >>> dev = qml.device('default.qubit', wires=2)
    >>> transformed_qfunc = qml.defer_measurements(qfunc)
    >>> qnode = qml.QNode(transformed_qfunc, dev)
    >>> par = np.array(np.pi/2, requires_grad=True)
    >>> qnode(par)
    tensor(0.43487747, requires_grad=True)

    We can also differentiate parameters passed to conditional operations:

    >>> qml.grad(qnode)(par)
    tensor(-0.49622252, requires_grad=True)

    Reusing and resetting measured wires will work as expected with the
    ``defer_measurements`` transform:

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def func(x, y):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            m_0 = qml.measure(1, reset=True)

            qml.cond(m_0, qml.RY)(y, wires=0)
            qml.RX(np.pi/4, wires=1)
            return qml.probs(wires=[0, 1])

    Executing this QNode:

    >>> pars = np.array([0.643, 0.246], requires_grad=True)
    >>> func(*pars)
    tensor([0.76960924, 0.13204407, 0.08394415, 0.01440254], requires_grad=True)

    .. details::
        :title: Usage Details

        By default, ``defer_measurements`` makes use of postselection information of
        mid-circuit measurements in the circuit in order to reduce the number of controlled
        operations and control wires. We can explicitly switch this feature off and compare
        the created circuits with and without this optimization. Consider the following circuit:

        .. code-block:: python3

            @qml.qnode(qml.device("default.qubit"))
            def node(x):
                qml.RX(x, 0)
                qml.RX(x, 1)
                qml.RX(x, 2)

                mcm0 = qml.measure(0, postselect=0, reset=False)
                mcm1 = qml.measure(1, postselect=None, reset=True)
                mcm2 = qml.measure(2, postselect=1, reset=False)
                qml.cond(mcm0+mcm1+mcm2==1, qml.RX)(0.5, 3)
                return qml.expval(qml.Z(0) @ qml.Z(3))

        Without the optimization, we find three gates controlled on the three measured
        qubits. They correspond to the combinations of controls that satisfy the condition
        ``mcm0+mcm1+mcm2==1``.

        >>> print(qml.draw(qml.defer_measurements(node, reduce_postselected=False))(0.6))
        0: ──RX(0.60)──|0⟩⟨0|─╭●─────────────────────────────────────────────┤ ╭<Z@Z>
        1: ──RX(0.60)─────────│──╭●─╭X───────────────────────────────────────┤ │
        2: ──RX(0.60)─────────│──│──│───|1⟩⟨1|─╭○────────╭○────────╭●────────┤ │
        3: ───────────────────│──│──│──────────├RX(0.50)─├RX(0.50)─├RX(0.50)─┤ ╰<Z@Z>
        4: ───────────────────╰X─│──│──────────├○────────├●────────├○────────┤
        5: ──────────────────────╰X─╰●─────────╰●────────╰○────────╰○────────┤

        If we do not explicitly deactivate the optimization, we obtain a much simpler circuit:

        >>> print(qml.draw(qml.defer_measurements(node))(0.6))
        0: ──RX(0.60)──|0⟩⟨0|─╭●─────────────────┤ ╭<Z@Z>
        1: ──RX(0.60)─────────│──╭●─╭X───────────┤ │
        2: ──RX(0.60)─────────│──│──│───|1⟩⟨1|───┤ │
        3: ───────────────────│──│──│──╭RX(0.50)─┤ ╰<Z@Z>
        4: ───────────────────╰X─│──│──│─────────┤
        5: ──────────────────────╰X─╰●─╰○────────┤

        There is only one controlled gate with only one control wire.

    .. details::
        :title: Deferred measurements with program capture

        ``qml.defer_measurements`` can be applied to callables when program capture is enabled. To do so,
        the ``aux_wires`` argument must be provided, which should be a sequence of integers to be used
        as the target wires for transforming mid-circuit measurements. With program capture enabled, some
        new features, as well as new restrictions are introduced, that are detailed below:

        **New features**

        * Arbitrary classical processing of mid-circuit measurement values is now possible. With
          program capture disabled, only limited classical processing, as detailed in the
          documentation for :func:`~pennylane.measure`. With program capture enabled, any ``jax.numpy``
          functions that can be applied to scalars can be used with mid-circuit measurements.

        * Using mid-circuit measurements as gate parameters is now possible. This feature currently
          has the following restrictions:
          * Mid-circuit measurement values cannot be used for multiple parameters of the same gate.
          * Mid-circuit measurement values cannot be used as wires.

          .. code-block:: python

              from functools import partial
              import jax
              import jax.numpy as jnp

              qml.capture.enable()

              @qml.capture.expand_plxpr_transforms
              @partial(qml.defer_measurements, aux_wires=list(range(5, 10)))
              def f():
                  m0 = qml.measure(0)

                  phi = jnp.sin(jnp.pi * m0)
                  qml.RX(phi, 0)
                  return qml.expval(qml.PauliZ(0))

          >>> jax.make_jaxpr(f)()
          { lambda ; . let
              _:AbstractOperator() = CNOT[n_wires=2] 0 5
              a:f64[] = mul 0.0 3.141592653589793
              b:f64[] = sin a
              c:AbstractOperator() = RX[n_wires=1] b 0
              _:AbstractOperator() = Controlled[
                control_values=(False,)
                work_wires=Wires([])
              ] c 5
              d:f64[] = mul 1.0 3.141592653589793
              e:f64[] = sin d
              f:AbstractOperator() = RX[n_wires=1] e 0
              _:AbstractOperator() = Controlled[
                control_values=(True,)
                work_wires=Wires([])
              ] f 5
              g:AbstractOperator() = PauliZ[n_wires=1] 0
              h:AbstractMeasurement(n_wires=None) = expval_obs g
            in (h,) }

        The above dummy example showcases how the transform is applied when the aforementioned
        features are used.

        **What doesn't work**

        * Currently, mid-circuit measurement values cannot be used in the condition for a
          :func:`~pennylane.while_loop`.
        * Currently, :func:`~pennylane.measure` cannot be used inside the body of control flow
          primitives. This includes :func:`~pennylane.cond`, :func:`~pennylane.while_loop`, and
          :func:`~pennylane.for_loop`.
        * Currently, if a branch of :func:`~pennylane.cond` uses mid-circuit measurements as its
          predicate, then all other branches must also use mid-circuit measurement values as
          predicates.
        * Currently, :func:`~pennylane.measure` cannot be used inside the body of functions
          being transformed with :func:`~pennylane.adjoint` or :func:`~pennylane.ctrl`.
    """
    if not any(isinstance(o, MidMeasureMP) for o in tape.operations):
        return (tape,), null_postprocessing

    _check_tape_validity(tape)

    new_operations = []

    # Find wires that are reused after measurement
    (
        measured_wires,
        reused_measurement_wires,
        any_repeated_measurements,
        is_postselecting,
    ) = _collect_mid_measure_info(tape)

    if is_postselecting and not allow_postselect:
        raise ValueError(
            "Postselection is not allowed on the device with deferred measurements. The device "
            "must support the Projector gate to apply postselection."
        )

    if len(reused_measurement_wires) > 0 and not all(isinstance(w, int) for w in tape.wires):
        raise ValueError(
            "qml.defer_measurements does not support custom wire labels with qubit reuse/reset."
        )

    # Apply controlled operations to store measurement outcomes and replace
    # classically controlled operations
    control_wires = {}
    cur_wire = (
        max(tape.wires) + 1 if reused_measurement_wires or any_repeated_measurements else None
    )

    for op in tape.operations:
        if isinstance(op, MidMeasureMP):
            _ = measured_wires.pop(0)

            if op.postselect is not None:
                with QueuingManager.stop_recording():
                    new_operations.append(qml.Projector([op.postselect], wires=op.wires[0]))

            # Store measurement outcome in new wire if wire gets reused
            if op.wires[0] in reused_measurement_wires or op.wires[0] in measured_wires:
                control_wires[op.id] = cur_wire

                with QueuingManager.stop_recording():
                    new_operations.append(qml.CNOT([op.wires[0], cur_wire]))

                if op.reset:
                    with QueuingManager.stop_recording():
                        # No need to manually reset if postselecting on |0>
                        if op.postselect is None:
                            new_operations.append(qml.CNOT([cur_wire, op.wires[0]]))
                        elif op.postselect == 1:
                            # We know that the measured wire will be in the |1> state if
                            # postselected |1>. So we can just apply a PauliX instead of
                            # a CNOT to reset
                            new_operations.append(qml.X(op.wires[0]))

                cur_wire += 1
            else:
                control_wires[op.id] = op.wires[0]

        elif op.__class__.__name__ == "Conditional":
            with QueuingManager.stop_recording():
                new_operations.extend(_add_control_gate(op, control_wires, reduce_postselected))
        else:
            new_operations.append(op)

    new_measurements = []

    for mp in tape.measurements:
        if mp.mv is not None:
            # Update measurement value wires. We can't use `qml.map_wires` because the same
            # wire can map to different control wires when multiple mid-circuit measurements
            # are made on the same wire. This mapping is determined by the id of the
            # MidMeasureMPs. Thus, we need to manually map wires for each MidMeasureMP.
            if isinstance(mp.mv, MeasurementValue):
                new_ms = [
                    qml.map_wires(m, {m.wires[0]: control_wires[m.id]}) for m in mp.mv.measurements
                ]
                new_m = MeasurementValue(new_ms, mp.mv.processing_fn)
            else:
                new_m = []
                for val in mp.mv:
                    new_ms = [
                        qml.map_wires(m, {m.wires[0]: control_wires[m.id]})
                        for m in val.measurements
                    ]
                    new_m.append(MeasurementValue(new_ms, val.processing_fn))

            with QueuingManager.stop_recording():
                new_mp = (
                    type(mp)(obs=new_m)
                    if not isinstance(mp, CountsMP)
                    else CountsMP(obs=new_m, all_outcomes=mp.all_outcomes)
                )
        else:
            new_mp = mp
        new_measurements.append(new_mp)

    new_tape = tape.copy(operations=new_operations, measurements=new_measurements)

    if is_postselecting and new_tape.batch_size is not None:
        # Split tapes if broadcasting with postselection
        return qml.transforms.broadcast_expand(new_tape)

    return [new_tape], null_postprocessing


def _add_control_gate(op, control_wires, reduce_postselected):
    """Helper function to add control gates"""
    if reduce_postselected:
        control = [control_wires[m.id] for m in op.meas_val.measurements if m.postselect is None]
        items = op.meas_val.postselected_items()
    else:
        control = [control_wires[m.id] for m in op.meas_val.measurements]
        items = op.meas_val.items()

    new_ops = []

    for branch, value in items:
        if value:
            # Empty sampling branches can occur when using _postselected_items
            new_op = (
                op.base
                if branch == ()
                else ctrl(op.base, control=Wires(control), control_values=branch)
            )
            new_ops.append(new_op)
    return new_ops
