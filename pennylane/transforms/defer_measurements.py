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

from functools import lru_cache, partial, reduce

import pennylane as qml
from pennylane.measurements import CountsMP, MeasurementValue, MidMeasureMP, ProbabilityMP, SampleMP
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
        from pennylane.capture.primitives import (
            AbstractMeasurement,
            AbstractOperator,
            cond_prim,
            ctrl_transform_prim,
            measure_prim,
        )
    except ImportError:
        return None, None

    class DeferMeasurementsInterpreter(PlxprInterpreter):
        """Interpreter for applying the defer_measurements transform to plxpr."""

        def __init__(self, num_wires):
            super().__init__()
            self._num_wires = num_wires
            self._measurements_map = {}

            # State variables
            self._cur_wire = None
            self._cur_measurement_idx = None

        def setup(self) -> None:
            """Initialize the instance before interpreting equations.

            Blank by default, this method can initialize any additional instance variables
            needed by an interpreter. For example, a device interpreter could initialize a statevector,
            or a compilation interpreter could initialize a staging area for the latest operation on each wire.
            """
            self._cur_wire = self._num_wires - 1
            self._cur_measurement_idx = 0

        def cleanup(self) -> None:
            """Perform any final steps after iterating through all equations.

            Blank by default, this method can clean up instance variables. Particularly,
            this method can be used to deallocate qubits and registers when converting to
            a Catalyst variant jaxpr.
            """
            self._measurements_map = {}
            self._cur_wire = None
            self._cur_measurement_idx = None

        def resolve_mcm_values(self, eqn, invals) -> MeasurementValue:
            """Create a ``MeasurementValue`` that captures all classical processing in its
            ``processing_fn``."""
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

                custom_handler = self._primitive_registrations.get(eqn.primitive, None)
                if custom_handler:
                    invals = [self.read(invar) for invar in eqn.invars]
                    outvals = custom_handler(self, *invals, **eqn.params)
                elif isinstance(eqn.outvars[0].aval, AbstractOperator):
                    outvals = self.interpret_operation_eqn(eqn)
                elif isinstance(eqn.outvars[0].aval, AbstractMeasurement):
                    outvals = self.interpret_measurement_eqn(eqn)
                else:
                    invals = [self.read(invar) for invar in eqn.invars]
                    if any(isinstance(inval, MeasurementValue) for inval in invals):
                        outvals = self.resolve_mcm_values(eqn, invals)
                    else:
                        outvals = eqn.primitive.bind(*invals, **eqn.params)

                if not eqn.primitive.multiple_results:
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
        with qml.QueuingManager.stop_recording():
            meas = type.__call__(
                MidMeasureMP,
                Wires(self._cur_wire),
                reset=reset,
                postselect=postselect,
                id=self._cur_measurement_idx,
            )

        cnot_wires = (wires, self._cur_wire)
        self._measurements_map[self._cur_measurement_idx] = self._cur_wire

        if postselect is not None:
            qml.Projector(jax.numpy.array([postselect]), wires=wires)

        qml.CNOT(wires=cnot_wires)

        if reset:
            if postselect is None:
                qml.CNOT(wires=cnot_wires[::-1])
            elif postselect == 1:
                qml.X(wires=wires)

        # cur_idx = self._cur_measurement_idx
        self._cur_measurement_idx += 1
        self._cur_wire -= 1
        return MeasurementValue([meas], lambda x: x)

    @DeferMeasurementsInterpreter.register_primitive(cond_prim)
    def _(self, *invals, jaxpr_branches, consts_slices, args_slice):
        n_branches = len(jaxpr_branches)
        conditions = invals[:n_branches]
        args = invals[args_slice]

        for i, (condition, jaxpr) in enumerate(zip(conditions, jaxpr_branches, strict=True)):
            if isinstance(condition, MeasurementValue):
                control_wires = Wires([m.wires[0] for m in condition.measurements])
                for branch, _ in condition._items():

                    # qml.capture.disable()
                    # value = condition.processing_fn(*branch)
                    # qml.capture.enable()
                    # if value:
                    #     cur_consts = invals[consts_slices[i]]
                    #     ctrl_transform_prim.bind(
                    #         *cur_consts,
                    #         *args,
                    #         *control_wires,
                    #         jaxpr=jaxpr,
                    #         n_control=len(control_wires),
                    #         control_values=branch,
                    #         work_wires=None,
                    #         n_consts=len(cur_consts),
                    #     )

                    value = condition.processing_fn(*branch)
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

        interpreter = DeferMeasurementsInterpreter()

        def wrapper(*inner_args):
            return interpreter.eval(jaxpr, consts, *inner_args)

        return jax.make_jaxpr(wrapper)(*args)

    return DeferMeasurementsInterpreter, defer_measurements_plxpr_to_plxpr


DeferMeasurementsInterpreter, defer_measurements_plxpr_to_plxpr = _get_plxpr_defer_measurements()


@partial(transform, plxpr_transform=defer_measurements_plxpr_to_plxpr)
def defer_measurements(
    tape: QuantumScript, reduce_postselected: bool = True, allow_postselect: bool = True
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
            of operations and control wires in the output tape. Active by default.
        allow_postselect (bool): Whether postselection is allowed. In order to perform postselection
            with ``defer_measurements``, the device must support the :class:`~.Projector` operation.
            Defaults to ``True``.

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
        items = op.meas_val._postselected_items()
    else:
        control = [control_wires[m.id] for m in op.meas_val.measurements]
        items = op.meas_val._items()

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
