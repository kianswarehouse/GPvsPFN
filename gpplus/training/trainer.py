import copy
import os
from typing import List, Optional

import gpytorch
import torch
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import get_settings, logger
from .callbacks import Callback
from .optimizers import LBFGSScipy
from .parameter_initializer import DefaultParameterInitializer, ParameterInitializer
from .stop_conditions import StopCondition
from .training_single_run import GPTrainerSingleProcess
from .initialization_prescreener import InitializationPrescreener


class GPTrainer:
    """
    GPTrainer handles the training process of a Gaussian Process model.

    Parameters:
        model (GPModel): The Gaussian Process model to train.
        optimizer_class (torch.optim.Optimizer, optional): The optimizer class to use for training.
        optimizer_kwargs (dict, optional): The arguments for the optimizer, excluding 'params'.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 50.
        seed (int, optional): Random seed for parameter initialization. Defaults to None.
        num_runs (int, optional): Number of runs (initializations). Defaults to 64.
        mll_class (gpytorch.mlls.MarginalLogLikelihood, optional): The Marginal Log Likelihood class to use.
        cholesky_jitter (float, optional): Jitter term for numerical stability in Cholesky. Defaults to 1e-6.
        callbacks (list[Callback]): Optional list of callback objects.
        stop_conditions (list[StopCondition], optional): List of stop conditions to check after each epoch.
            If None, defaults to ConvergencePatienceStopCondition(patience=20) and
            MinLossChangeStopCondition(min_loss_change=1e-7).
        device (str, optional): Device to run on. Defaults to "cpu", but set to "cuda" or "cuda:0"
                                if you have a GPU and want GPU training.
    """

    def __init__(
        self,
        model,
        optimizer_class: torch.optim.Optimizer = None,
        optimizer_kwargs: dict = None,
        scheduler_class: torch.optim.lr_scheduler.LRScheduler = None,
        scheduler_kwargs: dict = None,
        num_epochs: int = 50,
        seed: int = None,
        num_runs: int = 64,
        mll_class: gpytorch.mlls.MarginalLogLikelihood = None,
        cholesky_jitter: float = 1e-6,
        callbacks: Optional[List[Callback]] = None,
        initializer_class: ParameterInitializer = None,
        initializer_kwargs: dict = None,
        device: str = "cpu",
        stop_conditions: Optional[List[StopCondition]] = None,
        min_epochs: int = 0,
        # Prescreening parameters
        enable_prescreening: bool = False,
        num_test: Optional[int] = None,  # Total candidates to evaluate (default: num_runs * input_dim)
        prescreening_warmup_epochs: int = 3,
        prescreening_warmup_lr: float = 0.01,
        prescreening_optimizer_class=None,  # Optimizer for warmup (default: torch.optim.Adam)
        recorder=None,  # PrescreeningRecorder instance
    ):
        # -------------------------------------------------------
        # Set up the device (CPU or CUDA)
        # -------------------------------------------------------
        # If the user sets device="cuda" but CUDA is not available, fall back to CPU.
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        # Keep the original model on CPU
        self.model = model  # no .to(self.device) here
        logger.info("Model stays on CPU in the constructor.")

        # --------------------------------------------------
        #  CORE CONFIG
        # --------------------------------------------------
        self.num_epochs = num_epochs
        self.num_runs = num_runs
        self.seed = seed
        self.callbacks = callbacks or []
        self.cholesky_jitter = cholesky_jitter
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs
        self.min_epochs = min_epochs

        # Set default stop conditions if none provided
        if stop_conditions is None:
            from .stop_conditions import ConvergencePatienceStopCondition, MinLossChangeStopCondition

            self.stop_conditions = [
                ConvergencePatienceStopCondition(patience=20),
                MinLossChangeStopCondition(min_loss_change=1e-7),
            ]
        else:
            self.stop_conditions = stop_conditions
        # Get dtype from the model (which should be set from input data)
        if hasattr(model, "dtype") and model.dtype is not None:
            self.dtype = model.dtype
        else:
            self.dtype = torch.float64
            logger.warning(f"Model has no dtype attribute. Using {self.dtype} as fallback.")

        """
        # Initialize model parameters if requested
        if initialize_params:
            self.initialize_parameters(seed)
        """
        # Set up the initializer; use a default one if none is provided.
        if initializer_class is None:
            self.initializer = DefaultParameterInitializer(num_runs=self.num_runs, seed=self.seed)
        else:
            # Pass initializer_kwargs if provided, otherwise use empty dictionary
            if initializer_kwargs is None:
                initializer_kwargs = {}
            self.initializer = initializer_class(num_runs=self.num_runs, seed=self.seed, **initializer_kwargs)

        # Precompute number of parameters and Sobol samples.
        self.initializer.setup(model)

        # --------------------------------------------------
        #  OPTIMIZER
        # --------------------------------------------------
        # Handle optimizer class, use LBFGS as default
        if optimizer_class is None:
            self.optimizer_class = LBFGSScipy
            logger.warning("No optimizer class passed. Defaulting to LBFGS Scipy optimizer.")
        else:
            self.optimizer_class = optimizer_class

        # Handle optimizer arguments: set defaults per optimizer type when not provided
        if optimizer_kwargs is not None:
            self.optimizer_kwargs = optimizer_kwargs
        else:
            opt_cls = self.optimizer_class
            if opt_cls is LBFGSScipy or (
                hasattr(opt_cls, "__name__") and opt_cls.__name__ == "LBFGSScipy"
            ):
                self.optimizer_kwargs = {
                    "max_iter": 2000,
                    "max_eval": 5000,
                    "tolerance_grad": 1e-5,
                    "tolerance_change": 1e-9,
                    "history_size": 10,
                }
            elif opt_cls is torch.optim.Adam or (
                isinstance(opt_cls, type) and issubclass(opt_cls, torch.optim.Adam)
            ):
                self.optimizer_kwargs = {"lr": 0.01}
            else:
                self.optimizer_kwargs = {"max_iter": 20}
                logger.warning(
                    "No optimizer arguments passed and no built-in defaults for "
                    f"{getattr(opt_cls, '__name__', opt_cls)}. Using max_iter=20."
                )

        # Handle MLL class
        if mll_class is None:
            # Use the GPytorch MLL (marginal log likelihood) as the loss function
            self.mll_class = gpytorch.mlls.ExactMarginalLogLikelihood
            logger.warning("No MLL class passed. Defaulting to ExactMarginalLogLikelihood.")
        else:
            self.mll_class = mll_class
        
        # --------------------------------------------------
        #  PRESCREENING CONFIG
        # --------------------------------------------------
        self.enable_prescreening = enable_prescreening
        self.recorder = recorder
        self.prescreening_warmup_epochs = prescreening_warmup_epochs
        self.prescreening_warmup_lr = prescreening_warmup_lr
        self.prescreening_optimizer_class = prescreening_optimizer_class
        
        # Calculate num_test if not provided (default: num_runs * input_dim)
        if enable_prescreening:
            if num_test is None:
                # Try to get input_dim from model
                if hasattr(model, 'train_inputs') and model.train_inputs is not None and len(model.train_inputs) > 0:
                    input_dim = model.train_inputs[0].shape[1]
                elif hasattr(model, 'input_dim'):
                    input_dim = model.input_dim
                else:
                    # Fallback: use a reasonable default multiplier
                    input_dim = 10
                    logger.warning(f"Could not determine input_dim from model. Using default multiplier: {input_dim}")
                self.num_test = num_runs * input_dim
                logger.info(f"Prescreening: num_test not provided, using default: {num_runs} * {input_dim} = {self.num_test}")
            else:
                self.num_test = num_test
        else:
            self.num_test = None
        
        # Store selected indices from prescreening (will be set during train() if prescreening is enabled)
        self.selected_indices = None
        self.prescreen_initializer = None  # Will be set during train() if prescreening is enabled

    def train_single_process(self, run_index):
        """
        Runs training for a single initialization (run_index).
        - Copy the master CPU-based model
        - Initialize on CPU
        - Move the copy to GPU (if device is CUDA)
        - Train the copy
        - Return best loss + best state
        """
        # Copy the model (which is on CPU)
        base_model = copy.deepcopy(self.model)

        # Initialize parameters for the model copy on CPU using the initializer
        # If prescreening was used, use the prescreen_initializer with the selected index
        if self.prescreen_initializer is not None:
            self.prescreen_initializer.initialize(base_model, run_index)
        else:
            self.initializer.initialize(base_model, run_index)

        # Move model_copy to device
        base_model = base_model.to(self.device)

        # Train the model
        # Create isolated callback instances per run to avoid cross-run state mixing
        callbacks_copy = [copy.deepcopy(cb) for cb in self.callbacks] if self.callbacks else []
        # Set run_index on callbacks that support it
        for cb in callbacks_copy:
            if hasattr(cb, "set_run_index"):
                cb.set_run_index(run_index)
        # Create isolated stop condition instances per run to avoid cross-run state mixing
        stop_conditions_copy = [copy.deepcopy(sc) for sc in self.stop_conditions] if self.stop_conditions else None

        run = GPTrainerSingleProcess(
            model=base_model,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            mll_class=self.mll_class,
            num_epochs=self.num_epochs,
            cholesky_jitter=self.cholesky_jitter,
            callbacks=callbacks_copy,
            device=self.device,
            scheduler_class=self.scheduler_class,
            scheduler_kwargs=self.scheduler_kwargs,
            stop_conditions=stop_conditions_copy,
            min_epochs=self.min_epochs,
        )
        train_result = run.train()

        # Copy the trained parameters back to the original model
        # This ensures constraint enforcement is preserved
        with torch.no_grad():
            for (name, param), (_, trained_param) in zip(self.model.named_parameters(), base_model.named_parameters()):
                if param.requires_grad:
                    param.data.copy_(trained_param.data.to(dtype=param.dtype))

        # Collect callback data from callbacks_copy (they're deep-copied per run)
        callback_data = {}
        for cb in callbacks_copy:
            if hasattr(cb, 'get_stored_parameters'):
                cb_name = cb.__class__.__name__
                try:
                    stored_params = cb.get_stored_parameters()
                    # For JitterTrackingCallback, always include data even if empty list
                    # (empty list is falsy but we want to track that the callback ran)
                    if stored_params is not None:  # Changed from 'if stored_params' to handle empty lists
                        callback_data[cb_name] = stored_params
                except Exception as e:
                    logger.warning(f"Error getting stored parameters from {cb_name}: {e}")

        # Merge callback_data from train_result (if any) with callbacks_copy data
        if "callback_data" in train_result:
            # Merge dictionaries, with callbacks_copy taking precedence
            for key, value in train_result["callback_data"].items():
                if key not in callback_data:
                    callback_data[key] = value
        train_result["callback_data"] = callback_data

        return {"run_index": run_index, **train_result}

    def train_multiple_process_parallel(self, run_indices=None):
        """
        Train the model in parallel using different initialization runs.

        Args:
            run_indices: Optional list of run indices to use. If None, uses range(num_runs).
                        This is used when prescreening is enabled to train only selected initializations.

        Returns:
            list[dict]: A list of dictionaries containing training results
                        for each run (including error info if something fails).
        """

        # Use provided run_indices or default to range(num_runs)
        if run_indices is None:
            run_indices = list(range(self.num_runs))
        
        num_runs_to_train = len(run_indices)

        # defining a small wrapper to handle errors gracefully
        def safe_single_process(run_index, device_override=None):
            try:
                # Run the actual training job
                original_device = self.device
                if device_override is not None:
                    # Temporarily override the device for this run.
                    self.device = device_override
                _worker_init()
                result = self.train_single_process(run_index)
                # Restore the original device.
                self.device = original_device
                return result
            except Exception as e:
                # Log and return an error record for that run
                logger.exception(f"Error in training run #{run_index}: {e}")
                return {
                    "run_index": run_index,
                    "state_dict": None,
                    "loss": None,
                    "error": str(e),
                }

        def _worker_init():
            """Initialize worker process with global GP settings."""
            get_settings().apply()

        # Use joblib with appropriate backend to avoid time.sleep overhead
        # - CPU: Use 'loky' backend (multiprocessing) - eliminates time.sleep polling, true parallelism
        # - GPU: Use 'threading' backend with verbose=0 - CUDA doesn't work well with multiprocessing
        if self.device.type == "cpu":
            max_jobs = min(num_runs_to_train, max(1, (os.cpu_count() or 1) - 2))
            logger.info(
                f"Running {num_runs_to_train} runs using {max_jobs} parallel processes on CPU "
                f"(using joblib 'loky' backend - eliminates time.sleep polling)."
            )
            # Use loky backend (multiprocessing) - eliminates time.sleep polling, provides true parallelism
            # verbose=0 minimizes logging overhead
            results = Parallel(n_jobs=max_jobs, backend="loky", verbose=0)(
                delayed(safe_single_process)(run_index) for run_index in run_indices
            )

        elif str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
            num_gpus = torch.cuda.device_count()
            # Calculate total number of SMs (Streaming Multiprocessors) across all GPUs
            num_sms = sum(
                torch.cuda.get_device_properties(i).multi_processor_count
                for i in range(num_gpus)
            )
            # For GPU tasks, use threading backend (multiprocessing doesn't work well with CUDA)
            # Limit parallel jobs based on number of SMs
            max_jobs = min(num_runs_to_train, num_sms)
            logger.info(
                f"Running {num_runs_to_train} runs distributed across {num_gpus} GPUs "
                f"(using joblib 'threading' backend with verbose=0, max_jobs={max_jobs} based on {num_sms} total SMs)."
            )
            # Use threading backend for GPU (CUDA doesn't work with multiprocessing)
            # verbose=0 minimizes logging overhead (time.sleep still present but reduced)
            results = Parallel(n_jobs=max_jobs, backend="threading", verbose=0)(
                delayed(safe_single_process)(run_index, device_override=torch.device(f"cuda:{i % num_gpus}"))
                for i, run_index in enumerate(run_indices)
            )

        logger.info("Training completed.")
        return results
    
    def _aggregate_jitter_callbacks(self, results):
        """Aggregate jitter tracking data from all runs into a single file."""
        from .callbacks import JitterTrackingCallback
        
        # Find JitterTrackingCallback instances in callbacks
        jitter_callbacks = [cb for cb in (self.callbacks or []) if isinstance(cb, JitterTrackingCallback)]
        
        for jitter_cb in jitter_callbacks:
            if jitter_cb.save_file is not None:
                JitterTrackingCallback.aggregate_jitter_from_results(
                    results=results,
                    save_file=jitter_cb.save_file,
                    verbose=jitter_cb.verbose
                )
    
    def _aggregate_parameter_callbacks(self, results):
        """Aggregate parameter tracking data from all runs into a single file."""
        from .callbacks import IterationParameterCallback, EpochParameterCallback
        
        # Find IterationParameterCallback instances in callbacks
        iteration_callbacks = [cb for cb in (self.callbacks or []) if isinstance(cb, IterationParameterCallback)]
        
        for iter_cb in iteration_callbacks:
            if iter_cb.save_file is not None:
                IterationParameterCallback.aggregate_parameters_from_results(
                    results=results,
                    save_file=iter_cb.save_file,
                    verbose=iter_cb.verbose
                )
        
        # Find EpochParameterCallback instances in callbacks
        epoch_callbacks = [cb for cb in (self.callbacks or []) if isinstance(cb, EpochParameterCallback)]
        
        for epoch_cb in epoch_callbacks:
            if epoch_cb.save_file is not None:
                EpochParameterCallback.aggregate_parameters_from_results(
                    results=results,
                    save_file=epoch_cb.save_file,
                    verbose=epoch_cb.verbose
                )

    def train(self):
        # ------------------------------------------------------
        #  PRESCREENING (if enabled)
        # ------------------------------------------------------
        if self.enable_prescreening:
            logger.info("="*60)
            logger.info("PRESCREENING ENABLED")
            logger.info("="*60)
            
            # Create a temporary initializer for prescreening with num_test samples
            # We need to create a new initializer with num_runs=num_test
            # Use the same class and kwargs as the main initializer
            prescreen_initializer_class = self.initializer.__class__
            
            # Try to extract kwargs from the existing initializer if possible
            prescreen_initializer_kwargs = {}
            # Most initializers don't need special kwargs, but if they do, they should be passed via initializer_kwargs
            # For now, we'll just use the class with num_runs and seed
            
            prescreen_initializer = prescreen_initializer_class(
                num_runs=self.num_test,
                seed=self.seed,
                **prescreen_initializer_kwargs
            )
            prescreen_initializer.setup(self.model)
            
            # Create prescreener
            prescreener = InitializationPrescreener(
                model=self.model,
                initializer=prescreen_initializer,
                num_test=self.num_test,
                num_runs=self.num_runs,
                mll_class=self.mll_class,
                cholesky_jitter=self.cholesky_jitter,
                device=str(self.device),
                dtype=self.dtype,
                num_warmup_epochs=self.prescreening_warmup_epochs,
                warmup_lr=self.prescreening_warmup_lr,
                optimizer_class=self.prescreening_optimizer_class,
                recorder=self.recorder,
            )
            
            # Start recording if recorder is provided
            if self.recorder is not None:
                self.recorder.start_recording(self.num_test, self.num_runs)
            
            # Run prescreening to get selected indices
            self.selected_indices = prescreener.prescreen()
            
            logger.info(f"Prescreening complete. Selected {len(self.selected_indices)} initializations from {self.num_test} candidates.")
            logger.info(f"Selected indices (from prescreening): {self.selected_indices}")
            
            # Store the prescreen initializer for use during training
            # We'll use the selected indices with this initializer
            self.prescreen_initializer = prescreen_initializer
        else:
            self.selected_indices = None
        
        # ------------------------------------------------------
        #  TRAINING
        # ------------------------------------------------------
        # Call the multiple_process() method that trains using different initializations
        # If prescreening was enabled, use selected_indices; otherwise use range(num_runs)
        if self.selected_indices is not None:
            # Map selected indices to actual run indices
            # The selected_indices are from the prescreening (0 to num_test-1)
            # We need to use these indices when calling the initializer
            # But we also need to track which prescreening index corresponds to which training run
            run_indices = self.selected_indices
            logger.info(f"Training with prescreened initializations: {run_indices}")
        else:
            run_indices = None  # Will default to range(num_runs)
        
        results = self.train_multiple_process_parallel(run_indices=run_indices)
        
        # ------------------------------------------------------
        #  Aggregate jitter tracking data from all runs
        # ------------------------------------------------------
        self._aggregate_jitter_callbacks(results)
        
        # ------------------------------------------------------
        #  Aggregate parameter tracking data from all runs
        # ------------------------------------------------------
        self._aggregate_parameter_callbacks(results)

        # ------------------------------------------------------
        #  Select the best run by comparing the 'loss' values
        # ------------------------------------------------------
        best_run = None
        best_loss = float("inf")

        for run_result in results:
            if (
                run_result["loss"] is not None
                and run_result["loss"] < best_loss
                and run_result["state_dict"] is not None
            ):
                best_loss = run_result["loss"]
                best_run = run_result

        # ------------------------------------------------------
        #  If a valid best run was found, load it into self.model
        # ------------------------------------------------------
        if best_run is not None and best_run["state_dict"] is not None:
            state = best_run["state_dict"]
            # Load onto current model device (e.g. CPU); state may come from worker on different device
            device = next(self.model.parameters()).device
            state_cpu = {k: v.to(device) if hasattr(v, "to") else v for k, v in state.items()}
            self.model.load_state_dict(state_cpu)

            logger.info(
                f"Best run found: #{best_run['run_index']} with loss={best_loss:.4f}. "
                "Original model state_dict updated with best weights."
            )
            
            # Record final result in recorder if prescreening was enabled
            if self.enable_prescreening and self.recorder is not None:
                # Extract final parameters
                final_parameters = {}
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        final_parameters[name] = param.data.detach().cpu().clone()
                
                # Record the final result
                self.recorder.record_final_result(
                    selected_index=best_run['run_index'],
                    final_loss=best_loss,
                    final_parameters=final_parameters,
                )
        else:
            logger.warning("No valid best run found. Model was not updated.")

        return results