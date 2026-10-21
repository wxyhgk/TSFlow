"""UMA execution using the recovered calculator and numerical implementations."""

from uma_workflows.calculators.uma import UmaCalculatorFactory


class UmaExecutor(UmaCalculatorFactory):
    engine = "uma"
    method = "UMA"

    def execute(self, request, checkpoint, cancel_event, progress_callback=None):
        if request.calculation == "frequency_analysis":
            from app.adapters.uma.frequency import run_frequency_analysis

            return run_frequency_analysis(
                self, request, checkpoint, cancel_event, progress_callback
            )
        if request.calculation in {"single_point", "geometry_optimization"}:
            from app.adapters.uma.calculation import run_energy_or_optimization

            return run_energy_or_optimization(
                self, request, checkpoint, cancel_event, progress_callback
            )
        raise NotImplementedError("Transition-state workflow has not yet been restored")
