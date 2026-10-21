"""Map the standalone supplied-guess workflow to durable API results."""

import time
import numpy as np
from ase import Atoms
from app.contracts.jobs import (
    Atom,
    AtomicStructure,
    CalculationProgress,
    TransitionStateResult,
    SaddleOptimizationFrame,
    IrcPathFrameResult,
    IrcEndpointRelaxationResult,
    OptimizationFrame,
)
from app.formats.gaussian import render_transition_state_input
from uma_workflows.irc.workflow import BidirectionalIrcConfig
from uma_workflows.transition_states.supplied_guess import (
    SuppliedGuessConfig,
    run_supplied_guess,
)


def execute_supplied_ts(
    executor, request, checkpoint, cancel_event, progress_callback=None
):
    started = time.monotonic()
    p = request.parameters

    def to_atoms(structure):
        atoms = Atoms(
            [a.symbol for a in structure.atoms],
            positions=[a.position for a in structure.atoms],
            pbc=False,
        )
        atoms.info.update(charge=structure.charge, spin=structure.spin)
        return atoms

    def structure(positions):
        return AtomicStructure(
            atoms=[
                Atom(symbol=a.symbol, position=tuple(map(float, pos)))
                for a, pos in zip(request.structure.atoms, positions, strict=True)
            ],
            charge=request.structure.charge,
            spin=request.structure.spin,
        )

    calculator = executor.calculator(checkpoint, "cpu", request.task_name)
    analytic = None
    if p.hessian_method != "finite_difference":
        try:
            analytic = executor.hessian_calculator(
                checkpoint,
                request.task_name,
                use_vmap=len(request.structure.atoms) <= 30,
            )
        except Exception:
            if p.hessian_method == "analytic":
                raise
    reactant, product, guess = map(
        to_atoms,
        (request.structure, request.product_structure, request.transition_state_guess),
    )

    def progress(stage, step, limit, positions, energy, force):
        if progress_callback:
            progress_callback(
                CalculationProgress(
                    stage=stage,
                    step=min(step, limit),
                    max_steps=limit,
                    energy_ev=energy,
                    max_force_ev_per_a=force,
                    optimization_device="cpu",
                    current_structure=structure(positions),
                )
            )

    result = run_supplied_guess(
        reactant,
        product,
        guess,
        calculator=calculator,
        analytic_calculator=analytic,
        cancel_event=cancel_event,
        reaction_coordinates=request.reaction_coordinates,
        config=SuppliedGuessConfig(
            fmax=p.sella_cpu_fmax,
            max_steps=p.sella_max_steps,
            delta_a=p.frequency_delta_a,
            imaginary_threshold_cm1=p.imaginary_frequency_threshold_cm1,
            maximum_hessian_asymmetry=p.maximum_hessian_asymmetry,
            finite_difference_max_atoms=p.frequency_max_atoms,
            hessian_method=p.hessian_method,
            follow_irc=p.irc_follow,
            irc=BidirectionalIrcConfig(
                engine=p.irc_engine,
                steps=p.irc_steps,
                dx=p.irc_dx,
                inner_steps=p.irc_inner_steps,
                inner_fmax=p.irc_inner_fmax,
                fmax=p.irc_fmax,
                ts_fmax=p.sella_cpu_fmax,
                endpoint_distance_threshold_a=p.irc_endpoint_distance_threshold_a,
                require_endpoint_match=p.irc_require_endpoint_match,
                relax_endpoints=p.irc_relax_endpoints,
                endpoint_relaxation_fmax=p.irc_endpoint_relaxation_fmax,
                endpoint_relaxation_max_steps=p.irc_endpoint_relaxation_max_steps,
            ),
        ),
        on_saddle_frame=lambda f: progress(
            "geometric_cpu",
            f.step,
            p.sella_max_steps,
            f.positions_a,
            f.energy_ev,
            float(np.linalg.norm(f.forces_ev_per_a, axis=1).max()),
        ),
        on_irc_frame=lambda d, f: progress(
            "irc_" + d,
            f.step,
            p.irc_endpoint_relaxation_max_steps
            if d.endswith("_relaxation")
            else p.irc_steps,
            f.positions_a,
            f.energy_ev,
            f.max_force_ev_per_a,
        ),
    )
    energies = []
    for atoms in (reactant, product):
        atoms.calc = calculator
        energies.append(float(atoms.get_potential_energy()))
    screen = result.screening
    frequency = screen.frequency_analysis
    irc = result.irc
    final = structure(result.atoms.positions)
    extras = {}
    if irc:

        def relaxation(value):
            return IrcEndpointRelaxationResult(
                endpoint=value.endpoint,
                performed=value.performed,
                converged=value.converged,
                initial_energy_ev=value.initial_energy_ev,
                final_energy_ev=value.final_energy_ev,
                final_max_force_ev_per_a=value.final_max_force_ev_per_a,
                initial_structure=structure(value.initial_positions_a),
                relaxed_endpoint_structure=structure(value.final_positions_a),
                final_structure_label="relaxed_contact_minimum"
                if value.converged
                else "unconverged_relaxation_endpoint",
                trajectory=[
                    OptimizationFrame(
                        step=f.step,
                        energy_ev=f.energy_ev,
                        max_force_ev_per_a=f.max_force_ev_per_a,
                        structure=structure(f.positions_a),
                    )
                    for f in value.trajectory
                ],
                error=value.error,
            )

        extras = dict(
            irc_frames=[
                IrcPathFrameResult(
                    path_coordinate=f.path_coordinate,
                    energy_ev=f.energy_ev,
                    relative_energy_ev=f.relative_energy_ev,
                    max_force_ev_per_a=f.max_force_ev_per_a,
                    converged=f.converged,
                    structure=structure(f.positions_a),
                )
                for f in irc.frames
            ],
            irc_forward_frame_count=irc.forward_frame_count,
            irc_reverse_frame_count=irc.reverse_frame_count,
            irc_forward_inner_loop_failed=irc.forward_inner_loop_failed,
            irc_reverse_inner_loop_failed=irc.reverse_inner_loop_failed,
            irc_path_complete=irc.path_complete,
            irc_endpoint_distance_delta_a=irc.endpoint_distance_delta_a,
            irc_relaxed_endpoint_distance_delta_a=irc.relaxed_endpoint_distance_delta_a,
            irc_raw_endpoint_matches_expected=irc.raw_endpoint_topology.matches_expected_endpoints,
            irc_endpoint_matches_expected=irc.endpoint_topology.matches_expected_endpoints,
            irc_endpoint_assignment_final=result.validated,
            irc_endpoint_a_role=irc.endpoint_topology.endpoint_a_role
            if result.validated
            else None,
            irc_endpoint_b_role=irc.endpoint_topology.endpoint_b_role
            if result.validated
            else None,
            irc_endpoint_a_relaxation=relaxation(irc.endpoint_a_relaxation),
            irc_endpoint_b_relaxation=relaxation(irc.endpoint_b_relaxation),
        )
    return TransitionStateResult(
        transition_state=final,
        transition_state_energy_ev=screen.energy_ev,
        transition_state_forces_ev_per_a=screen.forces_ev_per_a.tolist(),
        transition_state_max_force_ev_per_a=screen.max_force_ev_per_a,
        reactant_energy_ev=energies[0],
        product_energy_ev=energies[1],
        forward_barrier_ev=screen.energy_ev - energies[0],
        reverse_barrier_ev=screen.energy_ev - energies[1],
        barrier_source="validated_transition_state"
        if result.validated
        else "legacy_candidate_energy",
        barrier_is_validated=result.validated,
        path=[],
        energy_profile=[],
        highest_path_coordinate=0.5,
        path_has_interior_maximum=False,
        dmf_iterations=0,
        converged=False,
        transition_state_stationary=screen.stationary,
        scientific_outcome="validated_transition_state"
        if result.validated
        else (
            "screened_first_order_saddle"
            if screen.pes_index1_saddle_valid
            else "unvalidated_candidate"
        ),
        saddle_refinement_performed=True,
        saddle_refinement_converged=result.refinement.converged,
        sella_cpu_steps=result.refinement.cpu_steps,
        saddle_trajectory=[
            SaddleOptimizationFrame(
                stage=f.stage,
                step=f.step,
                energy_ev=f.energy_ev,
                max_force_ev_per_a=float(
                    np.linalg.norm(f.forces_ev_per_a, axis=1).max()
                ),
                structure=structure(f.positions_a),
            )
            for f in result.refinement.frames
        ],
        frequency_analysis_performed=frequency is not None,
        frequency_scope=screen.frequency_scope,
        frequency_active_atom_indices=list(screen.frequency_active_atom_indices),
        frequencies_cm1=frequency.frequencies_cm1 if frequency else [],
        imaginary_frequencies_cm1=frequency.imaginary_frequencies_cm1
        if frequency
        else [],
        has_exactly_one_imaginary_mode=len(frequency.imaginary_frequencies_cm1) == 1
        if frequency
        else None,
        primary_imaginary_mode=frequency.primary_imaginary_mode.tolist()
        if frequency and frequency.primary_imaginary_mode is not None
        else None,
        hessian_method=frequency.hessian_method if frequency else None,
        hessian_asymmetry_relative=frequency.hessian_asymmetry_relative
        if frequency
        else None,
        rigid_body_projection_applied=frequency.rigid_body_projection_applied
        if frequency
        else None,
        rigid_mode_count=frequency.rigid_mode_count if frequency else None,
        hessian_quality_valid=screen.hessian_quality_valid,
        local_saddle_valid=screen.local_saddle_valid,
        irc_eligible=screen.irc_eligible,
        passes_uma_ts_screening=screen.passes_screening,
        screening_diagnostic_codes=list(screen.diagnostic_codes),
        screening_failure_code=screen.failure_code,
        irc_performed=result.irc_performed,
        irc_validation_passed=result.validated if p.irc_follow else None,
        irc_error=result.irc_error,
        optimizer_status="supplied_transition_state_guess",
        nmove=p.nmove,
        update_teval=p.update_teval,
        convergence=p.dmf_convergence,
        optimization_device="cpu",
        gaussian_input=render_transition_state_input(
            final,
            energy_ev=screen.energy_ev,
            max_force_ev_per_a=screen.max_force_ev_per_a,
        ),
        elapsed_seconds=time.monotonic() - started,
        warnings=list(result.warnings)
        + [
            "Supplied TS guess: no FB-ENM/DMF path search was run. Endpoint reference energies are single points, not relaxed asymptotes."
        ],
        **extras,
    )
