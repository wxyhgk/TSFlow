from __future__ import annotations

import io
import json
import time
from pathlib import Path
from threading import Event
from typing import Callable

import numpy as np

from uma_workflows.calculators import ANALYTIC_HESSIAN_VMAP_MAX_ATOMS
from uma_workflows.frequencies import numerical_frequency_analysis
from uma_workflows.irc import BidirectionalIrcConfig, run_bidirectional_irc
from uma_workflows.transition_states.workflow.models import SaddleConfig
from uma_workflows.transition_states.workflow.models import PathConfig
from uma_workflows.transition_states.workflow.path_bias import prepare_path_bias
from uma_workflows.transition_states.workflow.path_candidate import (
    optimize_path_candidate,
)
from uma_workflows.transition_states.workflow.path_curvature import (
    probe_path_curvature,
)
from uma_workflows.transition_states.workflow.saddle import refine_saddle_guess
from uma_workflows.transition_states import (
    CollectiveCoordinateTerm,
    FbenmConfig,
    PathBiasedCalculator,
    TransitionStateProblem,
    build_builtin_algorithm_registry,
    collective_coordinate_value_and_gradient,
    relaxed_collective_coordinate_scan,
)

from app.adapters.uma.structures import (
    atomic_structure_to_atoms,
    max_force,
    structure_fingerprint,
    structure_from_template,
)
from app.adapters.uma.path_artifacts import (
    load_optimized_path_artifact,
    load_path_candidate,
)
from app.contracts.action_results import (
    BidirectionalIrcActionResult,
    GeneratePathCandidatesActionResult,
    HessianModesActionResult,
    OptimizePathCandidateActionResult,
    PathCurvatureProbeSummary,
    PathCandidateSummary,
    ProbePathCurvatureActionResult,
    RunCollectiveCoordinateScanActionResult,
    SaddleRefinementActionResult,
)
from app.contracts.actions import (
    ComputeFullHessianAndModesAction,
    GeneratePathCandidatesAction,
    OptimizePathCandidateAction,
    ProbePathCurvatureAction,
    RefineSaddleAction,
    RunBidirectionalIrcAction,
    RunCollectiveCoordinateScanAction,
)
from app.contracts.calculations import CalculationProgress, OptimizationFrame
from app.contracts.artifacts import ArtifactReference
from app.contracts.path_artifacts import (
    CollectiveScanArtifactV1,
    CollectiveScanPointArtifactV1,
    OptimizedPathArtifactV1,
)
from app.contracts.transition_states import (
    CollectiveCoordinateDefinition,
    CollectiveCoordinateTermInput,
    CollectiveScanPointResult,
    EnergyProfilePoint,
    IrcEndpointRelaxationResult,
    IrcPathFrameResult,
)
from app.infrastructure.artifacts import FileArtifactStore
from app.application.actions.validation_profiles import UMA_TS_STRICT_V1


ProgressCallback = Callable[[CalculationProgress], None] | None


class UmaPrimitiveExecutor:
    def __init__(self, uma_executor, artifact_store: FileArtifactStore) -> None:
        self._uma = uma_executor
        self._artifacts = artifact_store

    def generate_path_candidates(
        self,
        execution_id,
        action: GeneratePathCandidatesAction,
        checkpoint: Path,
        cancel_event: Event,
        progress_callback: ProgressCallback,
    ) -> GeneratePathCandidatesActionResult:
        started = time.perf_counter()
        reactant = atomic_structure_to_atoms(action.structure)
        product = atomic_structure_to_atoms(action.product_structure)
        registry = build_builtin_algorithm_registry()
        initializer = registry.create_initializer(
            action.parameters.initializer_id,
            FbenmConfig(
                nmove=action.parameters.nmove,
                ensemble_size=action.parameters.ensemble_size,
                set_shared_graph_cell=(
                    action.parameters.evaluation_device == "mps"
                ),
            ),
        )
        calculator = self._uma._calculator(
            checkpoint,
            action.parameters.evaluation_device,
            action.task_name,
        )
        bias, bias_warnings = prepare_path_bias(
            reactant,
            product,
            tuple(action.reaction_coordinates),
            PathConfig(bias_mode=action.parameters.path_bias_mode),
        )
        problem = TransitionStateProblem(
            reactant=reactant,
            product=product,
            reaction_coordinates=tuple(action.reaction_coordinates),
        )

        def report(_stage, step, total, _snapshot) -> None:
            if progress_callback is None:
                return
            progress_callback(
                CalculationProgress(
                    step=step,
                    max_steps=total,
                    energy_ev=0.0,
                    max_force_ev_per_a=0.0,
                    optimization_device=action.parameters.evaluation_device,
                    current_structure=action.structure,
                    stage="fbenm",
                    reactant_endpoint=action.structure,
                    product_endpoint=action.product_structure,
                )
            )

        initialization = initializer.generate(
            problem,
            calculator=PathBiasedCalculator(calculator, bias),
            cancel_token=cancel_event,
            progress=report,
        )
        reactant_sha = structure_fingerprint(action.structure)
        product_sha = structure_fingerprint(action.product_structure)
        summaries: list[PathCandidateSummary] = []
        for ordinal, candidate in enumerate(initialization.candidates):
            positions = np.asarray(candidate.path.positions_a, dtype=float)
            candidate_index = int(
                candidate.candidate_index
                if candidate.candidate_index is not None
                else ordinal
            )
            payload = {
                "schema_version": 1,
                "model_name": action.model_name,
                "task_name": action.task_name,
                "reactant_sha256": reactant_sha,
                "product_sha256": product_sha,
                "atom_symbols": reactant.get_chemical_symbols(),
                "charge": action.structure.charge,
                "spin": action.structure.spin,
                "initializer_algorithm_id": initializer.metadata.algorithm_id,
                "initializer_algorithm_version": initializer.metadata.version,
                "candidate_index": candidate_index,
                "score_ev": candidate.score,
                "source": candidate.source,
                "path": candidate.path.to_dict(),
                "parameterization": (
                    candidate.parameterization.to_dict()
                    if candidate.parameterization is not None
                    else None
                ),
                "reaction_coordinates": [
                    coordinate.model_dump(mode="json")
                    for coordinate in action.reaction_coordinates
                ],
                "warnings": list(candidate.warnings),
                "metadata": dict(candidate.metadata),
            }
            reference = self._artifacts.write_json(
                execution_id,
                f"path-candidate-{ordinal}",
                "path_candidate",
                payload,
                frame_count=len(positions),
            )
            preview_atoms = reactant.copy()
            preview_atoms.set_positions(positions[len(positions) // 2])
            summaries.append(
                PathCandidateSummary(
                    candidate_index=candidate_index,
                    score_ev=candidate.score,
                    source=candidate.source,
                    frame_count=len(positions),
                    parameterization_kind=(
                        candidate.parameterization.kind
                        if candidate.parameterization is not None
                        else None
                    ),
                    preview_structure=structure_from_template(
                        action.structure,
                        preview_atoms,
                    ),
                    path_artifact=reference,
                    warnings=list(candidate.warnings),
                    metadata=dict(candidate.metadata),
                )
            )
        return GeneratePathCandidatesActionResult(
            initializer_algorithm_id=initializer.metadata.algorithm_id,
            initializer_algorithm_version=initializer.metadata.version,
            candidates=summaries,
            elapsed_seconds=time.perf_counter() - started,
            warnings=[*bias_warnings, *initialization.warnings],
        )

    def optimize_path_candidate(
        self,
        execution_id,
        action: OptimizePathCandidateAction,
        source_reference: ArtifactReference,
        checkpoint: Path,
        cancel_event: Event,
        progress_callback: ProgressCallback,
    ) -> OptimizePathCandidateActionResult:
        started = time.perf_counter()
        source_artifact, candidate = load_path_candidate(
            self._artifacts,
            source_reference,
            reactant=action.structure,
            product=action.product_structure,
            model_name=action.model_name,
            task_name=action.task_name,
            reaction_coordinates=action.reaction_coordinates,
        )
        reactant = atomic_structure_to_atoms(action.structure)
        product = atomic_structure_to_atoms(action.product_structure)
        calculator = self._uma._calculator(
            checkpoint,
            action.parameters.evaluation_device,
            action.task_name,
        )
        cpu_calculator = self._uma._calculator(
            checkpoint,
            "cpu",
            action.task_name,
        )
        problem = TransitionStateProblem(
            reactant=reactant,
            product=product,
            reaction_coordinates=tuple(action.reaction_coordinates),
        )

        def report(_stage, step, total, snapshot) -> None:
            if progress_callback is None or snapshot is None:
                return
            current = reactant.copy()
            current.set_positions(snapshot.path.positions_a[0])
            progress_callback(
                CalculationProgress(
                    step=step,
                    max_steps=total,
                    energy_ev=float(snapshot.energy_ev or 0.0),
                    max_force_ev_per_a=float(
                        snapshot.max_force_ev_per_a or 0.0
                    ),
                    optimization_device=action.parameters.evaluation_device,
                    current_structure=structure_from_template(
                        action.structure,
                        current,
                    ),
                    stage="dmf",
                    path_energies_ev=list(snapshot.path_energies_ev),
                    reactant_endpoint=action.structure,
                    product_endpoint=action.product_structure,
                )
            )

        result = optimize_path_candidate(
            problem,
            candidate,
            config=PathConfig(
                optimizer_id=action.parameters.optimizer_id,
                nmove=action.parameters.nmove,
                max_iterations=action.parameters.max_iterations,
                convergence=action.parameters.convergence,
                path_points=action.parameters.path_points,
                update_teval=action.parameters.update_teval,
                accept_every_trial_step=(
                    action.parameters.accept_every_trial_step
                ),
                bias_mode=action.parameters.path_bias_mode,
            ),
            device=action.parameters.evaluation_device,
            calculator=calculator,
            cpu_calculator=cpu_calculator,
            cancel_event=cancel_event,
            progress=report,
        )
        optimized_payload = OptimizedPathArtifactV1(
            model_name=action.model_name,
            task_name=action.task_name,
            reactant_sha256=source_artifact.reactant_sha256,
            product_sha256=source_artifact.product_sha256,
            source_path_sha256=source_reference.sha256,
            atom_symbols=source_artifact.atom_symbols,
            charge=source_artifact.charge,
            spin=source_artifact.spin,
            candidate_index=result.candidate_index,
            initializer_algorithm_id=(
                source_artifact.initializer_algorithm_id
            ),
            initializer_algorithm_version=(
                source_artifact.initializer_algorithm_version
            ),
            optimizer_algorithm_id=result.optimizer_algorithm_id,
            optimizer_algorithm_version=result.optimizer_algorithm_version,
            optimizer_parameters=action.parameters.model_dump(mode="json"),
            converged=result.optimization.converged,
            iterations=result.optimization.iterations,
            optimizer_status=result.optimization.optimizer_status,
            status_code=result.optimization.status_code,
            adaptive_sampling_enabled=(
                result.optimization.adaptive_sampling_enabled
            ),
            path=result.optimization.path.to_dict(),
            cpu_energies_ev=list(result.cpu_energies_ev),
            reaction_coordinates=action.reaction_coordinates,
            barrier_score_ev=result.score.barrier_score_ev,
            coordination_penalty_ev=result.score.coordination_penalty_ev,
            total_score_ev=result.score.total_score_ev,
            same_kind_progress_imbalance=(
                result.score.same_kind_progress_imbalance
            ),
            has_interior_maximum=result.has_interior_maximum,
            warnings=list(result.warnings),
        )
        optimized_reference = self._artifacts.write_json(
            execution_id,
            "optimized-path-candidate",
            "optimized_path_candidate",
            optimized_payload.model_dump(mode="json"),
            frame_count=len(result.optimization.path.positions_a),
        )
        highest_atoms = reactant.copy()
        highest_atoms.set_positions(result.highest_positions)
        baseline = result.cpu_energies_ev[0]
        energy_profile = [
            EnergyProfilePoint(
                path_coordinate=float(coordinate),
                energy_ev=energy,
                relative_energy_ev=energy - baseline,
            )
            for coordinate, energy in zip(
                result.sample_coordinates,
                result.cpu_energies_ev,
                strict=True,
            )
        ]
        return OptimizePathCandidateActionResult(
            source_path_artifact=source_reference,
            optimized_path_artifact=optimized_reference,
            candidate_index=result.candidate_index,
            optimizer_algorithm_id=result.optimizer_algorithm_id,
            optimizer_algorithm_version=result.optimizer_algorithm_version,
            converged=result.optimization.converged,
            iterations=result.optimization.iterations,
            optimizer_status=result.optimization.optimizer_status,
            status_code=result.optimization.status_code,
            adaptive_sampling_enabled=(
                result.optimization.adaptive_sampling_enabled
            ),
            energy_profile=energy_profile,
            highest_path_coordinate=result.highest_coordinate,
            highest_structure=structure_from_template(
                action.structure,
                highest_atoms,
            ),
            has_interior_maximum=result.has_interior_maximum,
            barrier_score_ev=result.score.barrier_score_ev,
            coordination_penalty_ev=result.score.coordination_penalty_ev,
            total_score_ev=result.score.total_score_ev,
            same_kind_progress_imbalance=(
                result.score.same_kind_progress_imbalance
            ),
            scientific_status=(
                "optimized_path_candidate"
                if result.optimization.converged
                else "path_not_converged"
            ),
            elapsed_seconds=time.perf_counter() - started,
            warnings=list(result.warnings),
        )

    def probe_path_curvature(
        self,
        execution_id,
        action: ProbePathCurvatureAction,
        source_reference: ArtifactReference,
        checkpoint: Path,
        cancel_event: Event,
        progress_callback: ProgressCallback,
    ) -> ProbePathCurvatureActionResult:
        del execution_id, progress_callback
        started = time.perf_counter()
        artifact = load_optimized_path_artifact(
            self._artifacts,
            source_reference,
            reactant=action.structure,
            product=action.product_structure,
            model_name=action.model_name,
            task_name=action.task_name,
            reaction_coordinates=action.reaction_coordinates,
        )
        atoms = atomic_structure_to_atoms(action.structure)
        calculator = self._uma._calculator(
            checkpoint,
            "cpu",
            action.task_name,
        )
        probed = probe_path_curvature(
            atoms,
            np.asarray(artifact.path.positions_a, dtype=float),
            np.asarray(artifact.path.coordinates, dtype=float),
            np.asarray(artifact.cpu_energies_ev, dtype=float),
            calculator=calculator,
            cancel_event=cancel_event,
            displacement_a=action.parameters.displacement_a,
            negative_threshold_ev_per_a2=(
                action.parameters.negative_threshold_ev_per_a2
            ),
            max_frames=action.parameters.max_frames,
        )
        summaries: list[PathCurvatureProbeSummary] = []
        selected_probe_index: int | None = None
        selected_structure: AtomicStructure | None = None
        for index, probe in enumerate(probed.probes):
            probe_atoms = atoms.copy()
            probe_atoms.set_positions(probe.positions_a)
            structure = structure_from_template(action.structure, probe_atoms)
            summaries.append(
                PathCurvatureProbeSummary(
                    frame_index=probe.frame_index,
                    path_coordinate=probe.path_coordinate,
                    structure=structure,
                    directional_curvature_ev_per_a2=(
                        probe.directional_curvature_ev_per_a2
                    ),
                    gradient_projection_ev_per_a=(
                        probe.gradient_projection_ev_per_a
                    ),
                    max_force_ev_per_a=probe.max_force_ev_per_a,
                    has_negative_directional_curvature=(
                        probe.has_negative_directional_curvature
                    ),
                )
            )
            if probe is probed.selected_probe:
                selected_probe_index = index
                selected_structure = structure
        return ProbePathCurvatureActionResult(
            source_optimized_path_artifact=source_reference,
            probes=summaries,
            peak_frame_index=probed.peak_frame_index,
            selected_probe_index=selected_probe_index,
            selected_structure=selected_structure,
            force_evaluations=probed.force_evaluations,
            scientific_status=(
                "negative_directional_curvature_seed"
                if probed.selected_probe is not None
                else "no_negative_directional_curvature"
            ),
            elapsed_seconds=time.perf_counter() - started,
            warnings=[
                "Directional curvature is a seed-screening diagnostic, not a "
                "normal-mode analysis or proof of an index-one saddle.",
                *probed.warnings,
            ],
        )

    def run_collective_coordinate_scan(
        self,
        execution_id,
        action: RunCollectiveCoordinateScanAction,
        source_reference: ArtifactReference,
        checkpoint: Path,
        cancel_event: Event,
        progress_callback: ProgressCallback,
    ) -> RunCollectiveCoordinateScanActionResult:
        started = time.perf_counter()
        artifact = load_optimized_path_artifact(
            self._artifacts,
            source_reference,
            reactant=action.structure,
            product=action.product_structure,
            model_name=action.model_name,
            task_name=action.task_name,
            reaction_coordinates=action.reaction_coordinates,
        )
        definition = _collective_coordinate_definition(action)
        terms = tuple(
            CollectiveCoordinateTerm(
                term.atom_i,
                term.atom_j,
                term.coefficient,
            )
            for term in definition.terms
        )
        path_positions = np.asarray(artifact.path.positions_a, dtype=float)
        path_values = np.asarray([
            collective_coordinate_value_and_gradient(frame, terms)[0]
            for frame in path_positions
        ])
        if action.parameters.target_values_a is not None:
            target_values = tuple(action.parameters.target_values_a)
        else:
            lower = float(np.min(path_values)) - action.parameters.range_padding_a
            upper = float(np.max(path_values)) + action.parameters.range_padding_a
            if upper - lower <= 1.0e-8:
                raise ValueError(
                    "optimized path has no progress along the collective coordinate"
                )
            target_values = tuple(
                float(value)
                for value in np.linspace(lower, upper, action.parameters.points)
            )
        peak_index = int(np.argmax(np.asarray(artifact.cpu_energies_ev)[1:-1])) + 1
        seed = atomic_structure_to_atoms(action.structure)
        seed.set_positions(path_positions[peak_index])
        calculator = self._uma._calculator(checkpoint, "cpu", action.task_name)

        def report(step: int, total: int, point) -> None:
            if progress_callback is None:
                return
            current = seed.copy()
            current.set_positions(point.positions_a)
            progress_callback(CalculationProgress(
                step=step,
                max_steps=total,
                energy_ev=point.energy_ev,
                max_force_ev_per_a=point.orthogonal_fmax_ev_per_a,
                optimization_device="cpu",
                current_structure=structure_from_template(action.structure, current),
                stage="collective_scan",
                reactant_endpoint=action.structure,
                product_endpoint=action.product_structure,
            ))

        scanned = relaxed_collective_coordinate_scan(
            seed,
            terms=terms,
            target_values_a=target_values,
            calculator=calculator,
            fmax=action.parameters.fmax,
            max_steps=action.parameters.max_steps_per_point,
            cancel_event=cancel_event,
            restraint_strength_ev_per_a2=(
                action.parameters.restraint_strength_ev_per_a2
            ),
            on_point=report,
        )
        baseline = min(point.energy_ev for point in scanned.points)
        point_results = [
            CollectiveScanPointResult(
                target_value_a=point.target_value_a,
                coordinate_value_a=point.coordinate_value_a,
                energy_ev=point.energy_ev,
                relative_energy_ev=point.energy_ev - baseline,
                derivative_ev_per_a=point.derivative_ev_per_a,
                orthogonal_fmax_ev_per_a=point.orthogonal_fmax_ev_per_a,
                converged=point.converged,
                steps=point.steps,
            )
            for point in scanned.points
        ]
        selected_index = (
            max(
                scanned.interior_maxima,
                key=lambda index: scanned.points[index].energy_ev,
            )
            if scanned.interior_maxima
            else None
        )
        warnings = []
        if any(not point.converged for point in scanned.points):
            warnings.append(
                "One or more restrained scan points did not converge; only "
                "maxima bracketed by converged neighbors are eligible seeds."
            )
        if selected_index is None:
            warnings.append(
                "The unbiased relaxed scan has no converged interior maximum."
            )
        scan_payload = CollectiveScanArtifactV1(
            model_name=action.model_name,
            task_name=action.task_name,
            reactant_sha256=artifact.reactant_sha256,
            product_sha256=artifact.product_sha256,
            source_optimized_path_sha256=source_reference.sha256,
            atom_symbols=artifact.atom_symbols,
            charge=artifact.charge,
            spin=artifact.spin,
            collective_coordinate=definition,
            points=[CollectiveScanPointArtifactV1(
                target_value_a=point.target_value_a,
                coordinate_value_a=point.coordinate_value_a,
                energy_ev=point.energy_ev,
                derivative_ev_per_a=point.derivative_ev_per_a,
                orthogonal_fmax_ev_per_a=point.orthogonal_fmax_ev_per_a,
                converged=point.converged,
                steps=point.steps,
                positions_a=point.positions_a.tolist(),
            ) for point in scanned.points],
            derivative_brackets=list(scanned.derivative_brackets),
            interior_maxima=list(scanned.interior_maxima),
            selected_maximum_index=selected_index,
            warnings=warnings,
        )
        scan_reference = self._artifacts.write_json(
            execution_id,
            "collective-coordinate-scan",
            "collective_coordinate_scan",
            scan_payload.model_dump(mode="json"),
            frame_count=len(scanned.points),
        )
        selected_structure = None
        if selected_index is not None:
            selected_atoms = seed.copy()
            selected_atoms.set_positions(scanned.points[selected_index].positions_a)
            selected_structure = structure_from_template(
                action.structure,
                selected_atoms,
            )
        if selected_index is not None:
            scientific_status = "collective_scan_seed"
        elif any(not point.converged for point in scanned.points):
            scientific_status = "collective_scan_unconverged"
        else:
            scientific_status = "collective_scan_no_interior_maximum"
        return RunCollectiveCoordinateScanActionResult(
            source_optimized_path_artifact=source_reference,
            scan_artifact=scan_reference,
            collective_coordinate=definition,
            points=point_results,
            derivative_brackets=list(scanned.derivative_brackets),
            interior_maxima=list(scanned.interior_maxima),
            selected_maximum_index=selected_index,
            selected_structure=selected_structure,
            scientific_status=scientific_status,
            elapsed_seconds=time.perf_counter() - started,
            warnings=warnings,
        )

    def refine_saddle(
        self,
        execution_id,
        action: RefineSaddleAction,
        checkpoint: Path,
        cancel_event: Event,
        progress_callback: ProgressCallback,
    ) -> SaddleRefinementActionResult:
        started = time.perf_counter()
        atoms = atomic_structure_to_atoms(action.structure)
        cpu_calculator = self._uma._calculator(
            checkpoint, "cpu", action.task_name
        )
        analytic = self._analytic_calculator(action, checkpoint)

        def report(progress) -> None:
            if progress_callback is None:
                return
            progress_callback(
                CalculationProgress(
                    step=progress.step,
                    max_steps=progress.max_steps,
                    energy_ev=progress.energy_ev,
                    max_force_ev_per_a=progress.max_force_ev_per_a,
                    optimization_device="cpu",
                    current_structure=structure_from_template(
                        action.structure,
                        _atoms_with_positions(atoms, progress.positions_a),
                    ),
                    stage="geometric_cpu",
                )
            )

        result = refine_saddle_guess(
            atoms,
            config=SaddleConfig(
                enabled=True,
                final_refinement_engine="geometric",
                cpu_refine=True,
                cpu_fmax=action.parameters.fmax,
                max_steps=action.parameters.max_steps,
            ),
            cpu_calculator=cpu_calculator,
            analytic_hessian_calculator=analytic,
            cancel_event=cancel_event,
            progress_callback=report,
            hessian_method=action.parameters.hessian_method,
            hessian_delta_a=action.parameters.hessian_delta_a,
            finite_difference_max_atoms=(
                action.parameters.finite_difference_max_atoms
            ),
        )
        result.atoms.calc = cpu_calculator
        energy = float(result.atoms.get_potential_energy())
        forces = np.asarray(result.atoms.get_forces(), dtype=float)
        fmax = max_force(forces)
        refinement = result.refinement
        trajectory = [
            OptimizationFrame(
                step=index,
                energy_ev=frame.energy_ev,
                max_force_ev_per_a=max_force(frame.forces_ev_per_a),
                structure=structure_from_template(
                    action.structure,
                    _atoms_with_positions(result.atoms, frame.positions_a),
                ),
            )
            for index, frame in enumerate(result.trajectory)
        ]
        return SaddleRefinementActionResult(
            structure=structure_from_template(action.structure, result.atoms),
            energy_ev=energy,
            forces_ev_per_a=_vectors(forces),
            max_force_ev_per_a=fmax,
            converged=bool(getattr(refinement, "converged", False)),
            stationary=fmax <= UMA_TS_STRICT_V1.stationary_fmax_ev_per_a,
            optimization_steps=len(trajectory),
            trajectory=trajectory,
            elapsed_seconds=time.perf_counter() - started,
            warnings=list(result.warnings),
        )

    def compute_full_hessian_and_modes(
        self,
        execution_id,
        action: ComputeFullHessianAndModesAction,
        checkpoint: Path,
        cancel_event: Event,
        progress_callback: ProgressCallback,
    ) -> HessianModesActionResult:
        started = time.perf_counter()
        atoms = atomic_structure_to_atoms(action.structure)
        analytic = self._analytic_calculator(action, checkpoint)
        cpu_calculator = self._uma._calculator(
            checkpoint, "cpu", action.task_name
        )
        atoms.calc = analytic or cpu_calculator
        energy = float(atoms.get_potential_energy())
        fmax = max_force(np.asarray(atoms.get_forces(), dtype=float))
        atom_count = len(atoms)
        exceeds_fd_limit = atom_count > action.parameters.finite_difference_max_atoms

        def report(coordinate: int, total: int) -> None:
            if progress_callback is not None:
                progress_callback(
                    CalculationProgress(
                        step=coordinate,
                        max_steps=total,
                        energy_ev=energy,
                        max_force_ev_per_a=fmax,
                        optimization_device="cpu",
                        current_structure=action.structure,
                        stage="frequencies",
                    )
                )

        analysis = numerical_frequency_analysis(
            atoms,
            calculator=cpu_calculator,
            analytic_calculator=analytic,
            hessian_method=action.parameters.hessian_method,
            delta_a=action.parameters.delta_a,
            imaginary_threshold_cm1=(
                UMA_TS_STRICT_V1.imaginary_frequency_threshold_cm1
            ),
            cancel_event=cancel_event,
            active_indices=list(range(atom_count)),
            allow_finite_difference_fallback=not exceeds_fd_limit,
            on_coordinate=report,
        )
        hessian_buffer = io.BytesIO()
        np.save(hessian_buffer, analysis.hessian_ev_per_a2, allow_pickle=False)
        hessian_ref = self._artifacts.write_bytes(
            execution_id,
            "full-hessian-npy",
            "full_hessian",
            "hessian.npy",
            hessian_buffer.getvalue(),
            media_type="application/x-npy",
        )
        hessian_quality = (
            analysis.hessian_asymmetry_relative
            <= UMA_TS_STRICT_V1.maximum_hessian_asymmetry
        )
        exactly_one = len(analysis.imaginary_frequencies_cm1) == 1
        stationary = fmax <= UMA_TS_STRICT_V1.stationary_fmax_ev_per_a
        pes_index1_saddle_valid = stationary and exactly_one and hessian_quality
        mode_payload = {
            "schema_version": 1,
            "atom_symbols": atoms.get_chemical_symbols(),
            "charge": action.structure.charge,
            "spin": action.structure.spin,
            "structure_sha256": structure_fingerprint(action.structure),
            "model_name": action.model_name,
            "task_name": action.task_name,
            "primary_imaginary_mode": (
                analysis.primary_imaginary_mode.tolist()
                if analysis.primary_imaginary_mode is not None
                else None
            ),
            "imaginary_frequencies_cm1": analysis.imaginary_frequencies_cm1,
            "hessian_sha256": hessian_ref.sha256,
            "pes_index1_saddle_valid": pes_index1_saddle_valid,
        }
        mode_ref = self._artifacts.write_json(
            execution_id,
            "normal-modes-json",
            "normal_modes",
            mode_payload,
        )
        warnings: list[str] = []
        if not stationary:
            warnings.append(
                "TS_NOT_STATIONARY: full Hessian was computed for diagnostics, "
                f"but Fmax={fmax:.6f} eV/A exceeds the validation profile."
            )
        if not exactly_one:
            warnings.append(
                "The full Hessian does not have exactly one significant imaginary mode."
            )
        if not hessian_quality:
            warnings.append("The Hessian symmetry residual exceeds the validation profile.")
        if analysis.hessian_fallback_reason:
            warnings.append(
                "Analytic Hessian fell back to finite differences: "
                f"{analysis.hessian_fallback_reason}"
            )
        return HessianModesActionResult(
            structure=action.structure,
            energy_ev=energy,
            max_force_ev_per_a=fmax,
            stationary=stationary,
            frequencies_cm1=analysis.frequencies_cm1,
            imaginary_frequencies_cm1=analysis.imaginary_frequencies_cm1,
            has_exactly_one_imaginary_mode=exactly_one,
            hessian_method=analysis.hessian_method,
            hessian_asymmetry_relative=analysis.hessian_asymmetry_relative,
            hessian_quality_valid=hessian_quality,
            pes_index1_saddle_valid=pes_index1_saddle_valid,
            hessian_artifact=hessian_ref,
            mode_artifact=mode_ref,
            elapsed_seconds=time.perf_counter() - started,
            warnings=warnings,
        )

    def run_bidirectional_irc(
        self,
        execution_id,
        action: RunBidirectionalIrcAction,
        checkpoint: Path,
        cancel_event: Event,
        progress_callback: ProgressCallback,
    ) -> BidirectionalIrcActionResult:
        started = time.perf_counter()
        if action.hessian_artifact.kind != "full_hessian":
            raise ValueError("run_bidirectional_irc requires a full_hessian artifact")
        if action.mode_artifact.kind != "normal_modes":
            raise ValueError("run_bidirectional_irc requires a normal_modes artifact")
        hessian = np.load(
            io.BytesIO(self._artifacts.read_verified(action.hessian_artifact)),
            allow_pickle=False,
        )
        mode_payload = json.loads(
            self._artifacts.read_verified(action.mode_artifact)
        )
        symbols = [atom.symbol for atom in action.structure.atoms]
        if mode_payload.get("atom_symbols") != symbols:
            raise ValueError("mode artifact atom ordering does not match the TS structure")
        if mode_payload.get("charge") != action.structure.charge or mode_payload.get(
            "spin"
        ) != action.structure.spin:
            raise ValueError("mode artifact charge/spin does not match the TS structure")
        if mode_payload.get("structure_sha256") != structure_fingerprint(
            action.structure
        ):
            raise ValueError("mode artifact TS coordinates do not match the TS structure")
        if mode_payload.get("model_name") != action.model_name or mode_payload.get(
            "task_name"
        ) != action.task_name:
            raise ValueError("mode artifact PES identity does not match the IRC action")
        if mode_payload.get("hessian_sha256") != action.hessian_artifact.sha256:
            raise ValueError("mode artifact is not bound to the supplied Hessian artifact")
        if mode_payload.get("pes_index1_saddle_valid") is not True:
            raise ValueError(
                "IRC requires artifacts produced by a validated full-Hessian "
                "index-1 saddle screen"
            )
        mode = mode_payload.get("primary_imaginary_mode")
        if mode is None:
            raise ValueError("mode artifact does not contain a primary imaginary mode")

        atoms = atomic_structure_to_atoms(action.structure)
        calculator = self._uma._calculator(checkpoint, "cpu", action.task_name)

        def report(direction, frame) -> None:
            if progress_callback is None:
                return
            progress_callback(
                CalculationProgress(
                    step=frame.step,
                    max_steps=(
                        action.parameters.endpoint_relaxation_max_steps
                        if direction.endswith("relaxation")
                        else action.parameters.steps
                    ),
                    energy_ev=frame.energy_ev,
                    max_force_ev_per_a=frame.max_force_ev_per_a,
                    optimization_device="cpu",
                    current_structure=structure_from_template(
                        action.structure,
                        _atoms_with_positions(atoms, frame.positions_a),
                    ),
                    stage=f"irc_{direction}",
                )
            )

        expected_reactant = (
            atomic_structure_to_atoms(action.expected_reactant)
            if action.expected_reactant is not None
            else None
        )
        expected_product = (
            atomic_structure_to_atoms(action.expected_product)
            if action.expected_product is not None
            else None
        )
        result = run_bidirectional_irc(
            atoms,
            hessian,
            validated_cartesian_mode=np.asarray(mode, dtype=float),
            calculator=calculator,
            expected_reactant=expected_reactant,
            expected_product=expected_product,
            config=BidirectionalIrcConfig(
                engine=action.parameters.engine,
                steps=action.parameters.steps,
                dx=action.parameters.dx,
                inner_steps=action.parameters.inner_steps,
                inner_fmax=min(
                    action.parameters.inner_fmax,
                    UMA_TS_STRICT_V1.irc_inner_fmax_ev_per_a,
                ),
                fmax=min(
                    action.parameters.fmax,
                    UMA_TS_STRICT_V1.irc_path_fmax_ev_per_a,
                ),
                ts_fmax=UMA_TS_STRICT_V1.stationary_fmax_ev_per_a,
                endpoint_distance_threshold_a=(
                    action.parameters.endpoint_distance_threshold_a
                ),
                require_endpoint_match=expected_reactant is not None,
                relax_endpoints=action.parameters.relax_endpoints,
                endpoint_relaxation_fmax=min(
                    action.parameters.endpoint_relaxation_fmax,
                    UMA_TS_STRICT_V1.irc_endpoint_fmax_ev_per_a,
                ),
                endpoint_relaxation_max_steps=(
                    action.parameters.endpoint_relaxation_max_steps
                ),
            ),
            cancel_event=cancel_event,
            on_frame=report,
        )
        frames = [
            IrcPathFrameResult(
                path_coordinate=frame.path_coordinate,
                energy_ev=frame.energy_ev,
                relative_energy_ev=frame.relative_energy_ev,
                max_force_ev_per_a=frame.max_force_ev_per_a,
                converged=frame.converged,
                structure=structure_from_template(
                    action.structure,
                    _atoms_with_positions(atoms, frame.positions_a),
                ),
            )
            for frame in result.frames
        ]
        endpoint_a = _endpoint_result(
            action.structure, atoms, result.endpoint_a_relaxation
        )
        endpoint_b = _endpoint_result(
            action.structure, atoms, result.endpoint_b_relaxation
        )
        topology = result.endpoint_topology
        return BidirectionalIrcActionResult(
            transition_state=action.structure,
            frames=frames,
            forward_frame_count=result.forward_frame_count,
            reverse_frame_count=result.reverse_frame_count,
            path_complete=result.path_complete,
            endpoint_a=endpoint_a.relaxed_endpoint_structure,
            endpoint_b=endpoint_b.relaxed_endpoint_structure,
            endpoint_a_relaxation=endpoint_a,
            endpoint_b_relaxation=endpoint_b,
            endpoint_match_evaluated=topology is not None,
            endpoint_matches_expected=(
                topology.matches_expected_endpoints if topology is not None else None
            ),
            endpoint_a_role=topology.endpoint_a_role if topology is not None else None,
            endpoint_b_role=topology.endpoint_b_role if topology is not None else None,
            elapsed_seconds=time.perf_counter() - started,
            warnings=list(result.warnings),
        )

    def _analytic_calculator(self, action, checkpoint: Path):
        if action.parameters.hessian_method == "finite_difference":
            return None
        try:
            return self._uma.hessian_calculator(
                checkpoint,
                action.task_name,
                use_vmap=(
                    len(action.structure.atoms)
                    <= ANALYTIC_HESSIAN_VMAP_MAX_ATOMS
                ),
            )
        except Exception:
            if action.parameters.hessian_method == "analytic":
                raise
            return None


def _collective_coordinate_definition(
    action: RunCollectiveCoordinateScanAction,
) -> CollectiveCoordinateDefinition:
    if action.collective_coordinate is not None:
        return action.collective_coordinate
    terms = [
        CollectiveCoordinateTermInput(
            atom_i=coordinate.atom_i,
            atom_j=coordinate.atom_j,
            coefficient=(1.0 if coordinate.kind == "breaking" else -1.0),
            label=coordinate.label or coordinate.kind,
        )
        for coordinate in action.reaction_coordinates
    ]
    return CollectiveCoordinateDefinition(terms=terms)


def _atoms_with_positions(template, positions):
    atoms = template.copy()
    atoms.set_positions(np.asarray(positions, dtype=float))
    return atoms


def _vectors(values) -> list[tuple[float, float, float]]:
    return [tuple(float(value) for value in row) for row in np.asarray(values)]


def _endpoint_result(template, atoms, relaxation) -> IrcEndpointRelaxationResult:
    initial = structure_from_template(
        template,
        _atoms_with_positions(atoms, relaxation.initial_positions_a),
    )
    final = structure_from_template(
        template,
        _atoms_with_positions(atoms, relaxation.final_positions_a),
    )
    trajectory = [
        OptimizationFrame(
            step=frame.step,
            energy_ev=frame.energy_ev,
            max_force_ev_per_a=frame.max_force_ev_per_a,
            structure=structure_from_template(
                template,
                _atoms_with_positions(atoms, frame.positions_a),
            ),
        )
        for frame in relaxation.trajectory
    ]
    return IrcEndpointRelaxationResult(
        endpoint=relaxation.endpoint,
        performed=relaxation.performed,
        converged=relaxation.converged,
        initial_energy_ev=relaxation.initial_energy_ev,
        final_energy_ev=relaxation.final_energy_ev,
        final_max_force_ev_per_a=relaxation.final_max_force_ev_per_a,
        initial_structure=initial,
        relaxed_endpoint_structure=final,
        final_structure_label=(
            "relaxed_contact_minimum"
            if relaxation.converged
            else "unconverged_relaxation_endpoint"
        ),
        trajectory=trajectory,
        error=relaxation.error,
    )
