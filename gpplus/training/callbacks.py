from abc import ABC
from typing import Any, TypedDict


class CallbackOnEpochStartContext(TypedDict):
    epoch: int
    model: Any
    trainer: Any
    device: str


class CallbackOnEpochEndContext(TypedDict):
    epoch: int
    model: Any
    trainer: Any
    loss: float
    device: str


class CallbackOnTrainStartContext(TypedDict):
    model: Any
    trainer: Any
    device: str


class CallbackOnTrainEndContext(TypedDict):
    epoch: int
    model: Any
    trainer: Any
    best_loss: float
    best_state_dict: Any
    device: str


class Callback(ABC):
    def on_epoch_start(self, context: CallbackOnEpochStartContext):
        """
        Called at the start of each epoch during training.

        Args:
            context (dict): A dictionary containing training state info.
        """
        pass

    def on_epoch_end(self, context: CallbackOnEpochEndContext):
        """
        Called at the end of each epoch during training.

        Args:
            context (dict): A dictionary containing training state info.
        """
        pass

    def on_train_start(self, context: CallbackOnTrainStartContext):
        """
        Called at the start of each training.

        Args:
            context (dict): A dictionary containing training state info.
        """
        pass

    def on_train_end(self, context: CallbackOnTrainEndContext):
        """
        Called at the end of each training.

        Args:
            context (dict): A dictionary containing training state info.
        """
        pass


class PrintLossCallback(Callback):
    def on_epoch_end(self, context: dict):
        print(f"Epoch {context['epoch']} - Loss: {context['loss']:.4f}")

class PrintInitialParametersCallback(Callback):
    def on_train_start(self, context: dict):
        print("Initial parameters: ")
        for name, param in context["model"].named_parameters():
            print(name, param.data)


class FinalParameterStorageCallback(Callback):
    """
    Callback to store final raw parameters (raw_lengthscales, raw_outputscales, raw_noise) 
    from each model after training for later reference in results files.
    
    Args:
        save_file (str): Path to save the parameters JSON file
        verbose (bool): Whether to print parameter values when storing
    """
    
    def __init__(self, save_file: str = "final_parameters.json", verbose: bool = True):
        self.save_file = save_file
        self.verbose = verbose
        self.stored_parameters = []
        self._initial_params = None
        self._run_count = 0
        self._best_epoch = 0  # Track best epoch when loss improves
        self._current_best_loss = float("inf")
    
    def on_train_start(self, context: dict):
        """Capture initial parameters at the start of training."""
        model = context["model"]
        self._run_count += 1
        self._best_epoch = 0
        self._current_best_loss = float("inf")
        try:
            # Use epoch 0 for initial snapshot (epoch may be unavailable here)
            self._initial_params = self._extract_final_parameters(model, epoch=0, best_loss=None)
            if self.verbose:
                print(f"\n=== Initial Parameters (Run {self._run_count}, Epoch 0) ===")
                print(f"Raw noise: {self._initial_params['raw_noise']}")
                print(f"Raw outputscale: {self._initial_params['raw_outputscale']}")
                raw_lengthscales = self._initial_params.get('raw_lengthscales', [])
                if raw_lengthscales:
                    print(f"Raw lengthscales: {raw_lengthscales} (count: {len(raw_lengthscales)})")
                else:
                    print(f"Raw lengthscales: N/A (not found)")
                raw_cat_lengthscales = self._initial_params.get('raw_cat_lengthscales', [])
                if raw_cat_lengthscales:
                    print(f"Raw cat lengthscales: {raw_cat_lengthscales} (count: {len(raw_cat_lengthscales)})")
                raw_source_lengthscales = self._initial_params.get('raw_source_lengthscales', [])
                if raw_source_lengthscales:
                    print(f"Raw source lengthscales: {raw_source_lengthscales} (count: {len(raw_source_lengthscales)})")
        except Exception as e:
            print(f"Error capturing initial parameters: {e}")
            import traceback
            traceback.print_exc()
    
    def on_epoch_end(self, context: dict):
        """Track best epoch when loss improves."""
        loss = context.get("loss")
        epoch = context.get("epoch", 0)
        if loss is not None and loss < self._current_best_loss:
            self._current_best_loss = loss
            self._best_epoch = epoch
    
    def on_train_end(self, context: dict):
        """Store final parameters at the end of training."""
        model = context["model"]
        epoch = context["epoch"]
        best_loss = context.get("best_loss", None)
        trainer = context.get("trainer", None)
        # Get cholesky_jitter from trainer if available
        cholesky_jitter = None
        if trainer is not None and hasattr(trainer, 'cholesky_jitter'):
            cholesky_jitter = trainer.cholesky_jitter
        best_state_dict = context.get("best_state_dict", None)
        
        try:
            # If best_state_dict is available, load it temporarily to extract parameters from best model
            # Otherwise, use the current model state
            if best_state_dict is not None:
                import copy
                temp_model = copy.deepcopy(model)
                temp_model.load_state_dict(best_state_dict)
                model_to_extract = temp_model
            else:
                model_to_extract = model
            
            # Extract parameters (including transformed ones)
            final_params = self._extract_final_parameters(
                model_to_extract, 
                epoch, 
                best_loss, 
                cholesky_jitter,
                self._best_epoch
            )
            record = self._combine_initial_final(self._initial_params, final_params)
            self.stored_parameters.append(record)
            
            if self.verbose:
                print(f"\n=== Final Parameters (Run {self._run_count}, Epoch {epoch}) ===")
                print(f"Number of epochs: {final_params.get('num_epochs', 'N/A')}")
                print(f"Best epoch: {final_params.get('best_epoch', 'N/A')}")
                print(f"Jitter: {final_params.get('jitter', 'N/A')}")
                print(f"Noise (transformed): {final_params.get('noise', 'N/A')}")
                print(f"Outputscale (transformed): {final_params.get('outputscale', 'N/A')}")
                lengthscales = final_params.get('lengthscales', [])
                if lengthscales:
                    print(f"Lengthscales (transformed): {lengthscales} (count: {len(lengthscales)})")
                else:
                    print(f"Lengthscales (transformed): N/A (not found)")
                
                # Show cat_kernel lengthscales if they exist
                cat_lengthscales = final_params.get('cat_lengthscales', [])
                if cat_lengthscales:
                    print(f"Cat lengthscales (transformed): {cat_lengthscales} (count: {len(cat_lengthscales)})")
                
                # Show source_kernel lengthscales if they exist
                source_lengthscales = final_params.get('source_lengthscales', [])
                if source_lengthscales:
                    print(f"Source lengthscales (transformed): {source_lengthscales} (count: {len(source_lengthscales)})")
                
                print(f"Raw noise: {final_params['raw_noise']}")
                print(f"Raw outputscale: {final_params['raw_outputscale']}")
                raw_lengthscales = final_params.get('raw_lengthscales', [])
                if raw_lengthscales:
                    print(f"Raw lengthscales: {raw_lengthscales} (count: {len(raw_lengthscales)})")
                else:
                    print(f"Raw lengthscales: N/A (not found)")
                
                # Show raw cat_kernel and source_kernel lengthscales if they exist
                raw_cat_lengthscales = final_params.get('raw_cat_lengthscales', [])
                if raw_cat_lengthscales:
                    print(f"Raw cat lengthscales: {raw_cat_lengthscales} (count: {len(raw_cat_lengthscales)})")
                
                raw_source_lengthscales = final_params.get('raw_source_lengthscales', [])
                if raw_source_lengthscales:
                    print(f"Raw source lengthscales: {raw_source_lengthscales} (count: {len(raw_source_lengthscales)})")
                print(f"Best loss: {best_loss}")
                # If we have initial, also print deltas
                if self._initial_params is not None:
                    deltas = record.get('deltas', {})
                    print("--- Deltas (final - initial) ---")
                    print(f"Delta raw_noise: {deltas.get('raw_noise')}")
                    print(f"Delta raw_outputscale: {deltas.get('raw_outputscale')}")
                    print(f"Delta raw_lengthscales: {deltas.get('raw_lengthscales')}")
            
            # Save to file
            # self._save_parameters()
            
        except Exception as e:
            print(f"Error storing final parameters: {e}")
            import traceback
            traceback.print_exc()

    def _combine_initial_final(self, initial: Any, final: dict) -> dict:
        """Combine initial and final snapshots and compute deltas.

        If no initial snapshot is available, returns a record with only final values.
        """
        record = {
            "run": self._run_count,
            "epoch": final.get("epoch"),
            "num_epochs": final.get("num_epochs"),
            "best_epoch": final.get("best_epoch"),
            "best_loss": final.get("best_loss"),
            "jitter": final.get("jitter"),
            "timestamp": final.get("timestamp"),
            "initial": None,
            "final": None,
            "deltas": None,
        }

        record["final"] = {
            "raw_noise": final.get("raw_noise"),
            "raw_outputscale": final.get("raw_outputscale"),
            "raw_lengthscales": final.get("raw_lengthscales"),
            "raw_cat_lengthscales": final.get("raw_cat_lengthscales"),
            "raw_source_lengthscales": final.get("raw_source_lengthscales"),
            "noise": final.get("noise"),  # Transformed
            "outputscale": final.get("outputscale"),  # Transformed
            "lengthscales": final.get("lengthscales"),  # Transformed (from cont_kernel)
            "cat_lengthscales": final.get("cat_lengthscales"),  # Transformed (from cat_kernel)
            "source_lengthscales": final.get("source_lengthscales"),  # Transformed (from source_kernel)
            "kernel_type": final.get("kernel_type"),
            "input_dim": final.get("input_dim"),
        }

        if initial is None:
            return record

        record["initial"] = {
            "raw_noise": initial.get("raw_noise"),
            "raw_outputscale": initial.get("raw_outputscale"),
            "raw_lengthscales": initial.get("raw_lengthscales"),
            "raw_cat_lengthscales": initial.get("raw_cat_lengthscales"),
            "raw_source_lengthscales": initial.get("raw_source_lengthscales"),
            "kernel_type": initial.get("kernel_type"),
            "input_dim": initial.get("input_dim"),
        }

        # Compute deltas (final - initial)
        try:
            noise_init = initial.get("raw_noise")
            noise_final = final.get("raw_noise")
            outscale_init = initial.get("raw_outputscale")
            outscale_final = final.get("raw_outputscale")
            ls_init = initial.get("raw_lengthscales") or []
            ls_final = final.get("raw_lengthscales") or []

            noise_delta = None if (noise_init is None or noise_final is None) else float(noise_final - noise_init)
            outscale_delta = None if (outscale_init is None or outscale_final is None) else float(outscale_final - outscale_init)

            # Align lengths for lengthscales
            max_len = max(len(ls_init), len(ls_final))
            ls_init_aligned = list(ls_init) + [0.0] * (max_len - len(ls_init))
            ls_final_aligned = list(ls_final) + [0.0] * (max_len - len(ls_final))
            ls_delta = [float(f - i) for f, i in zip(ls_final_aligned, ls_init_aligned)] if max_len > 0 else []

            record["deltas"] = {
                "raw_noise": noise_delta,
                "raw_outputscale": outscale_delta,
                "raw_lengthscales": ls_delta,
            }
        except Exception:
            record["deltas"] = None

        return record
    
    def _extract_final_parameters(self, model, epoch: int, best_loss: float = None, cholesky_jitter: float = None, best_epoch: int = None):
        """Extract raw and transformed parameters from the model using recursive search."""
        params = {
            "epoch": epoch,
            "num_epochs": epoch + 1,  # +1 because epochs are 0-indexed
            "best_epoch": best_epoch if best_epoch is not None else epoch,
            "best_loss": best_loss,
            "jitter": cholesky_jitter,
            "timestamp": None,
            "raw_noise": None,
            "raw_outputscale": None,
            "raw_lengthscales": None,
            "noise": None,  # Transformed
            "outputscale": None,  # Transformed
            "lengthscales": [],  # Transformed
            "kernel_type": None,
            "input_dim": None
        }
        
        # Add timestamp
        from datetime import datetime
        params["timestamp"] = datetime.now().isoformat()
        
        # Recursively search for raw parameters
        noise_params = []
        outputscale_params = []
        lengthscale_params = []
        
        self._recursive_parameter_search(model, noise_params, outputscale_params, lengthscale_params)
        
        # Extract the first found parameter of each type (raw)
        if noise_params:
            params['raw_noise'] = noise_params[0]
        
        if outputscale_params:
            params['raw_outputscale'] = outputscale_params[0]
        
        # Always check base_kernel for raw lengthscales - it's the authoritative source
        # This ensures we get all lengthscales for ARD kernels
        if hasattr(model, 'covar_module') and hasattr(model.covar_module, 'base_kernel'):
            base_kernel = model.covar_module.base_kernel
            
            # Special handling for CombinedKernel - extract from cont_kernel, cat_kernel, and source_kernel
            if hasattr(base_kernel, 'cont_kernel') and base_kernel.cont_kernel is not None:
                # Extract from cont_kernel (trainable lengthscales)
                cont_kernel = base_kernel.cont_kernel
                if hasattr(cont_kernel, 'raw_lengthscale'):
                    try:
                        ls = cont_kernel.raw_lengthscale.data.flatten().tolist()
                        if ls:
                            params['raw_lengthscales'] = ls
                    except Exception:
                        pass
                
                # Extract from cat_kernel (fixed at 0)
                if hasattr(base_kernel, 'cat_kernel') and base_kernel.cat_kernel is not None:
                    cat_kernel = base_kernel.cat_kernel
                    # Check if it's a single kernel or a list (MultCatKs)
                    if isinstance(cat_kernel, list):
                        # MultCatKs - extract from all cat_kernels
                        all_cat_ls = []
                        for idx, ck in enumerate(cat_kernel):
                            if hasattr(ck, 'raw_lengthscale'):
                                try:
                                    cat_ls = ck.raw_lengthscale.data.flatten().tolist()
                                    if cat_ls:
                                        all_cat_ls.extend(cat_ls)
                                except Exception:
                                    pass
                        if all_cat_ls:
                            params['raw_cat_lengthscales'] = all_cat_ls
                    else:
                        # Single cat_kernel
                        if hasattr(cat_kernel, 'raw_lengthscale'):
                            try:
                                cat_ls = cat_kernel.raw_lengthscale.data.flatten().tolist()
                                if cat_ls:
                                    params['raw_cat_lengthscales'] = cat_ls
                            except Exception:
                                pass
                
                # Extract from source_kernel (fixed at 0)
                if hasattr(base_kernel, 'source_kernel') and base_kernel.source_kernel is not None:
                    source_kernel = base_kernel.source_kernel
                    if hasattr(source_kernel, 'raw_lengthscale'):
                        try:
                            source_ls = source_kernel.raw_lengthscale.data.flatten().tolist()
                            if source_ls:
                                params['raw_source_lengthscales'] = source_ls
                        except Exception:
                            pass
            # Special handling for CombinedKernel_MultCatKs (no cont_kernel, but has multiple cat_kernels)
            elif hasattr(base_kernel, 'cat_kernel'):
                cat_kernel_attr = base_kernel.cat_kernel
                # Check if cat_kernel is a list (MultCatKs) or if we have cat_kernel_0, cat_kernel_1, etc.
                if isinstance(cat_kernel_attr, list) or hasattr(base_kernel, 'cat_kernel_0'):
                    # This is CombinedKernel_MultCatKs
                    all_cat_ls = []
                    
                    # Get list of cat_kernels
                    if isinstance(cat_kernel_attr, list):
                        cat_kernels = cat_kernel_attr
                    else:
                        # Extract all cat_kernel_* modules
                        cat_kernels = []
                        i = 0
                        while hasattr(base_kernel, f'cat_kernel_{i}'):
                            cat_kernels.append(getattr(base_kernel, f'cat_kernel_{i}'))
                            i += 1
                    
                    # Extract raw lengthscales from each cat_kernel
                    for ck in cat_kernels:
                        if hasattr(ck, 'raw_lengthscale'):
                            try:
                                cat_ls = ck.raw_lengthscale.data.flatten().tolist()
                                if cat_ls:
                                    all_cat_ls.extend(cat_ls)
                            except Exception:
                                pass
                    
                    if all_cat_ls:
                        params['raw_lengthscales'] = all_cat_ls
            elif hasattr(base_kernel, 'raw_lengthscale'):
                try:
                    ls = base_kernel.raw_lengthscale.data.flatten().tolist()
                    if ls:
                        params['raw_lengthscales'] = ls
                except Exception:
                    pass
        
        # Fallback to recursive search results if base_kernel didn't have raw_lengthscale
        if not params.get('raw_lengthscales') and lengthscale_params:
            params['raw_lengthscales'] = lengthscale_params
        
        # Extract transformed parameters
        self._extract_transformed_parameters(model, params)
        
        # Try to determine kernel type and input dimension
        params['kernel_type'] = self._determine_kernel_type(model)
        if hasattr(model, 'train_inputs') and model.train_inputs:
            params['input_dim'] = model.train_inputs[0].shape[-1]
        
        return params
    
    def _extract_transformed_parameters(self, model, params):
        """Extract transformed parameters (not raw) from the model."""
        # Track whether we've already captured the continuous lengthscales
        params.setdefault("_lengthscales_locked", False)

        # Extract transformed noise
        try:
            if hasattr(model, 'likelihood') and hasattr(model.likelihood, 'noise'):
                noise = model.likelihood.noise
                if noise.numel() == 1:
                    params["noise"] = float(noise.item())
                else:
                    params["noise"] = noise.detach().cpu().numpy().flatten().tolist()
        except Exception:
            pass
        
        # Recursively extract transformed outputscale and lengthscales
        self._recursive_extract_transformed_kernel_params(model, params)
        
        # ALWAYS check base_kernel directly for ARD lengthscales and override any previous results
        # This is the authoritative source for ARD kernels wrapped in LogScaleKernel
        # The recursive search might find lengthscales from wrapper kernels, but we want the base_kernel ones
        if hasattr(model, 'covar_module'):
            covar = model.covar_module

            def _has_component(kernel_obj):
                return any(
                    hasattr(kernel_obj, attr) and getattr(kernel_obj, attr) is not None
                    for attr in ('cont_kernel', 'cat_kernel', 'source_kernel')
                )

            base_kernel_attr = covar.base_kernel if hasattr(covar, 'base_kernel') else None
            combined_kernel = None
            if base_kernel_attr is not None and _has_component(base_kernel_attr):
                combined_kernel = base_kernel_attr
            if combined_kernel is None and _has_component(covar):
                combined_kernel = covar
            
            if combined_kernel is not None:
                # Special handling for combined kernels - extract from cont_kernel, cat_kernel, and source_kernel
                if hasattr(combined_kernel, 'cont_kernel') and combined_kernel.cont_kernel is not None:
                    cont_kernel = combined_kernel.cont_kernel
                    try:
                        ls_list = None
                        # Always use raw_lengthscale to ensure we get all ARD dimensions
                        if hasattr(cont_kernel, 'raw_lengthscale'):
                            raw_ls = cont_kernel.raw_lengthscale
                            if raw_ls is not None and raw_ls.numel() > 0:
                                if hasattr(cont_kernel, 'raw_lengthscale_constraint'):
                                    constraint = cont_kernel.raw_lengthscale_constraint
                                    transformed = constraint.transform(raw_ls)
                                    ls_list = transformed.detach().cpu().numpy().flatten().tolist()
                                else:
                                    ls_list = raw_ls.detach().cpu().numpy().flatten().tolist()
                        
                        # Fallback to lengthscale property if raw_lengthscale didn't work
                        if ls_list is None and hasattr(cont_kernel, 'lengthscale'):
                            try:
                                ls = cont_kernel.lengthscale
                                if ls is not None and ls.numel() > 0:
                                    ls_list = ls.detach().cpu().numpy().flatten().tolist()
                            except (AttributeError, RuntimeError):
                                pass
                        
                        if ls_list is not None:
                            params["lengthscales"] = ls_list
                            params["_lengthscales_locked"] = True
                            if self.verbose:
                                print(f"[DEBUG] Extracted {len(ls_list)} lengthscales from cont_kernel (shape: {raw_ls.shape if 'raw_ls' in locals() else 'unknown'}): {ls_list[:5]}..." if len(ls_list) > 5 else f"[DEBUG] Extracted {len(ls_list)} lengthscales from cont_kernel: {ls_list}")
                    except Exception as e:
                        import logging
                        logging.debug(f"Error extracting lengthscales from cont_kernel: {e}")
                    
                    # Extract from cat_kernel (should be fixed at 0)
                    if hasattr(combined_kernel, 'cat_kernel') and combined_kernel.cat_kernel is not None:
                        cat_kernel = combined_kernel.cat_kernel
                        try:
                            cat_ls_list = None
                            # Always use raw_lengthscale to ensure we get all ARD dimensions
                            if hasattr(cat_kernel, 'raw_lengthscale'):
                                cat_raw_ls = cat_kernel.raw_lengthscale
                                if cat_raw_ls is not None and cat_raw_ls.numel() > 0:
                                    if hasattr(cat_kernel, 'raw_lengthscale_constraint'):
                                        constraint = cat_kernel.raw_lengthscale_constraint
                                        transformed = constraint.transform(cat_raw_ls)
                                        cat_ls_list = transformed.detach().cpu().numpy().flatten().tolist()
                                    else:
                                        cat_ls_list = cat_raw_ls.detach().cpu().numpy().flatten().tolist()
                            
                            # Fallback to lengthscale property if raw_lengthscale didn't work
                            if cat_ls_list is None and hasattr(cat_kernel, 'lengthscale'):
                                try:
                                    cat_ls = cat_kernel.lengthscale
                                    if cat_ls is not None and cat_ls.numel() > 0:
                                        cat_ls_list = cat_ls.detach().cpu().numpy().flatten().tolist()
                                except (AttributeError, RuntimeError):
                                    pass
                            
                            if cat_ls_list is not None:
                                params["cat_lengthscales"] = cat_ls_list
                                if self.verbose:
                                    print(f"[DEBUG] Extracted {len(cat_ls_list)} lengthscales from cat_kernel (shape: {cat_raw_ls.shape if 'cat_raw_ls' in locals() else 'unknown'}): {cat_ls_list[:5]}..." if len(cat_ls_list) > 5 else f"[DEBUG] Extracted {len(cat_ls_list)} lengthscales from cat_kernel: {cat_ls_list}")
                        except Exception as e:
                            import logging
                            logging.debug(f"Error extracting lengthscales from cat_kernel: {e}")
                    
                    # Extract from source_kernel (should be fixed at 0)
                    if hasattr(combined_kernel, 'source_kernel') and combined_kernel.source_kernel is not None:
                        source_kernel = combined_kernel.source_kernel
                        try:
                            source_ls_list = None
                            # Always use raw_lengthscale to ensure we get all ARD dimensions
                            if hasattr(source_kernel, 'raw_lengthscale'):
                                source_raw_ls = source_kernel.raw_lengthscale
                                if source_raw_ls is not None and source_raw_ls.numel() > 0:
                                    if hasattr(source_kernel, 'raw_lengthscale_constraint'):
                                        constraint = source_kernel.raw_lengthscale_constraint
                                        transformed = constraint.transform(source_raw_ls)
                                        source_ls_list = transformed.detach().cpu().numpy().flatten().tolist()
                                    else:
                                        source_ls_list = source_raw_ls.detach().cpu().numpy().flatten().tolist()
                            
                            # Fallback to lengthscale property if raw_lengthscale didn't work
                            if source_ls_list is None and hasattr(source_kernel, 'lengthscale'):
                                try:
                                    source_ls = source_kernel.lengthscale
                                    if source_ls is not None and source_ls.numel() > 0:
                                        source_ls_list = source_ls.detach().cpu().numpy().flatten().tolist()
                                except (AttributeError, RuntimeError):
                                    pass
                            
                            if source_ls_list is not None:
                                params["source_lengthscales"] = source_ls_list
                                if self.verbose:
                                    print(f"[DEBUG] Extracted {len(source_ls_list)} lengthscales from source_kernel (shape: {source_raw_ls.shape if 'source_raw_ls' in locals() else 'unknown'}): {source_ls_list[:5]}..." if len(source_ls_list) > 5 else f"[DEBUG] Extracted {len(source_ls_list)} lengthscales from source_kernel: {source_ls_list}")
                        except Exception as e:
                            import logging
                            logging.debug(f"Error extracting lengthscales from source_kernel: {e}")
                
                # Special handling for CombinedKernel_MultCatKs - extract from all cat_kernel_* modules
                # Check if this is CombinedKernel_MultCatKs by looking for multiple cat_kernel modules
                if hasattr(combined_kernel, 'cat_kernel'):
                    cat_kernel_attr = combined_kernel.cat_kernel
                    # Check if cat_kernel is a list (MultCatKs) or if we have cat_kernel_0, cat_kernel_1, etc.
                    if isinstance(cat_kernel_attr, list) or hasattr(combined_kernel, 'cat_kernel_0'):
                        # This is CombinedKernel_MultCatKs
                        all_cat_lengthscales = []
                        
                        # Get list of cat_kernels
                        if isinstance(cat_kernel_attr, list):
                            cat_kernels = cat_kernel_attr
                        else:
                            # Extract all cat_kernel_* modules
                            cat_kernels = []
                            i = 0
                            while hasattr(combined_kernel, f'cat_kernel_{i}'):
                                cat_kernels.append(getattr(combined_kernel, f'cat_kernel_{i}'))
                                i += 1
                        
                        # Extract lengthscales from each cat_kernel
                        for idx, cat_kernel in enumerate(cat_kernels):
                            try:
                                cat_ls_list = None
                                # Always use raw_lengthscale to ensure we get all ARD dimensions
                                if hasattr(cat_kernel, 'raw_lengthscale'):
                                    cat_raw_ls = cat_kernel.raw_lengthscale
                                    if cat_raw_ls is not None and cat_raw_ls.numel() > 0:
                                        if hasattr(cat_kernel, 'raw_lengthscale_constraint'):
                                            constraint = cat_kernel.raw_lengthscale_constraint
                                            transformed = constraint.transform(cat_raw_ls)
                                            cat_ls_list = transformed.detach().cpu().numpy().flatten().tolist()
                                        else:
                                            cat_ls_list = cat_raw_ls.detach().cpu().numpy().flatten().tolist()
                                
                                # Fallback to lengthscale property if raw_lengthscale didn't work
                                if cat_ls_list is None and hasattr(cat_kernel, 'lengthscale'):
                                    try:
                                        cat_ls = cat_kernel.lengthscale
                                        if cat_ls is not None and cat_ls.numel() > 0:
                                            cat_ls_list = cat_ls.detach().cpu().numpy().flatten().tolist()
                                    except (AttributeError, RuntimeError):
                                        pass
                                
                                if cat_ls_list is not None:
                                    all_cat_lengthscales.extend(cat_ls_list)
                                    if self.verbose:
                                        print(f"[DEBUG] Extracted {len(cat_ls_list)} lengthscales from cat_kernel_{idx} (shape: {cat_raw_ls.shape if 'cat_raw_ls' in locals() else 'unknown'}): {cat_ls_list[:5]}..." if len(cat_ls_list) > 5 else f"[DEBUG] Extracted {len(cat_ls_list)} lengthscales from cat_kernel_{idx}: {cat_ls_list}")
                            except Exception as e:
                                import logging
                                logging.debug(f"Error extracting lengthscales from cat_kernel_{idx}: {e}")
                        
                        # Store all cat lengthscales
                        if all_cat_lengthscales:
                            params["cat_lengthscales"] = all_cat_lengthscales
                            if self.verbose:
                                print(f"[DEBUG] Extracted {len(all_cat_lengthscales)} total cat lengthscales from {len(cat_kernels)} cat_kernels: {all_cat_lengthscales[:10]}..." if len(all_cat_lengthscales) > 10 else f"[DEBUG] Extracted {len(all_cat_lengthscales)} total cat lengthscales from {len(cat_kernels)} cat_kernels: {all_cat_lengthscales}")
                
            # If we haven't extracted lengthscales yet, try regular kernel extraction
            fallback_kernel = base_kernel_attr if base_kernel_attr is not None else covar
            if "lengthscales" not in params or params["lengthscales"] is None:
                try:
                    ls_list = None
                    if hasattr(fallback_kernel, 'lengthscale'):
                        try:
                            ls = fallback_kernel.lengthscale
                            if ls is not None and ls.numel() > 0:
                                ls_list = ls.detach().cpu().numpy().flatten().tolist()
                        except (AttributeError, RuntimeError):
                            pass
                    
                    if ls_list is None and hasattr(fallback_kernel, 'raw_lengthscale'):
                        raw_ls = fallback_kernel.raw_lengthscale
                        if raw_ls is not None and raw_ls.numel() > 0:
                            if hasattr(fallback_kernel, 'raw_lengthscale_constraint'):
                                constraint = fallback_kernel.raw_lengthscale_constraint
                                transformed = constraint.transform(raw_ls)
                                ls_list = transformed.detach().cpu().numpy().flatten().tolist()
                            else:
                                ls_list = raw_ls.detach().cpu().numpy().flatten().tolist()
                    
                    if ls_list is not None:
                        params["lengthscales"] = ls_list
                        if self.verbose:
                            print(f"[DEBUG] Extracted {len(ls_list)} lengthscales from base_kernel: {ls_list[:5]}..." if len(ls_list) > 5 else f"[DEBUG] Extracted {len(ls_list)} lengthscales from base_kernel: {ls_list}")
                except Exception as e:
                    import logging
                    logging.debug(f"Error extracting lengthscales from base_kernel: {e}")
                    pass

        params.pop("_lengthscales_locked", None)
    
    def _recursive_extract_transformed_kernel_params(self, obj, params, visited=None, depth=0):
        """Recursively search through model components for transformed kernel parameters."""
        if visited is None:
            visited = set()
        
        # Avoid infinite recursion and limit depth
        obj_id = id(obj)
        if obj_id in visited or depth > 10:
            return
        visited.add(obj_id)
        
        try:
            # Extract transformed outputscale
            if hasattr(obj, 'outputscale') and params["outputscale"] is None:
                try:
                    outputscale = obj.outputscale
                    if outputscale.numel() == 1:
                        params["outputscale"] = float(outputscale.item())
                    else:
                        params["outputscale"] = outputscale.detach().cpu().numpy().flatten().tolist()
                except:
                    pass
            
            # Extract transformed lengthscales
            # Skip lengthscale extraction here - we'll get it directly from base_kernel in the fallback
            # This avoids picking up lengthscales from the wrong kernel components
            if hasattr(obj, 'lengthscale') and not hasattr(obj, 'base_kernel'):
                # Only extract if this is a base kernel (doesn't have base_kernel attribute)
                # This ensures we get the ARD kernel's lengthscales, not from wrapper kernels
                try:
                    lengthscale = obj.lengthscale
                    if lengthscale.numel() > 0:
                        lengthscale_list = lengthscale.detach().cpu().numpy().flatten().tolist()
                        # Only use if this has multiple lengthscales (ARD) or we haven't found any yet
                        if len(lengthscale_list) > 1:
                            # This is an ARD kernel, use it
                            if not params.get("_lengthscales_locked"):
                                params["lengthscales"] = lengthscale_list
                                params["_lengthscales_locked"] = True
                        elif not params.get("lengthscales"):
                            # No lengthscales found yet, use these (might be isotropic)
                            if not params.get("_lengthscales_locked"):
                                params["lengthscales"] = lengthscale_list
                except Exception as e:
                    # Silently continue if extraction fails for this object
                    pass
            
            # Recursively search through specific attributes that are likely to contain kernels/parameters
            # Search base_kernel first to prioritize ARD kernel lengthscales
            search_attrs = [
                'base_kernel',  # Search base_kernel first for ARD kernels
                'covar_module', 'likelihood', 'mean_module', 
                'cat_kernel', 'cont_kernel', 'source_kernel', 'kernels',
                'noise_covar', 'cat_encoder', 'source_encoder'
            ]
            
            for attr_name in search_attrs:
                if hasattr(obj, attr_name):
                    try:
                        attr = getattr(obj, attr_name)
                        if attr is not None:
                            self._recursive_extract_transformed_kernel_params(attr, params, visited, depth + 1)
                    except:
                        continue
            
            # Also search through registered modules
            if hasattr(obj, '_modules'):
                for module_name, module in obj._modules.items():
                    if module is not None:
                        self._recursive_extract_transformed_kernel_params(module, params, visited, depth + 1)
                        
        except Exception:
            pass
    
    def _recursive_parameter_search(self, obj, noise_params, outputscale_params, lengthscale_params, visited=None, depth=0):
        """Recursively search through model components for raw parameters."""
        if visited is None:
            visited = set()
        
        # Avoid infinite recursion and limit depth
        obj_id = id(obj)
        if obj_id in visited or depth > 10:
            return
        visited.add(obj_id)
        
        try:
            # Check if this object has the parameters we're looking for
            if hasattr(obj, 'raw_noise'):
                try:
                    if hasattr(obj.raw_noise, 'data') and obj.raw_noise.data is not None:
                        noise_val = obj.raw_noise.data.item()
                        noise_params.append(noise_val)
                except:
                    pass
            
            if hasattr(obj, 'raw_outputscale'):
                try:
                    if hasattr(obj.raw_outputscale, 'data') and obj.raw_outputscale.data is not None:
                        outputscale_val = obj.raw_outputscale.data.item()
                        outputscale_params.append(outputscale_val)
                except:
                    pass
            
            if hasattr(obj, 'raw_lengthscale'):
                try:
                    if hasattr(obj.raw_lengthscale, 'data') and obj.raw_lengthscale.data is not None:
                        lengthscale_val = obj.raw_lengthscale.data.flatten().tolist()
                        lengthscale_params.extend(lengthscale_val)
                except:
                    pass
            
            # Recursively search through specific attributes that are likely to contain kernels/parameters
            search_attrs = [
                'covar_module', 'likelihood', 'mean_module', 'base_kernel', 
                'cat_kernel', 'cont_kernel', 'source_kernel', 'kernels',
                'noise_covar', 'cat_encoder', 'source_encoder'
            ]
            
            for attr_name in search_attrs:
                if hasattr(obj, attr_name):
                    try:
                        attr = getattr(obj, attr_name)
                        if attr is not None:
                            self._recursive_parameter_search(attr, noise_params, outputscale_params, lengthscale_params, visited, depth + 1)
                    except:
                        continue
            
            # Also search through registered modules
            if hasattr(obj, '_modules'):
                for module_name, module in obj._modules.items():
                    if module is not None:
                        self._recursive_parameter_search(module, noise_params, outputscale_params, lengthscale_params, visited, depth + 1)
                        
        except Exception:
            pass
    
    def _determine_kernel_type(self, model):
        """Determine the kernel type by examining the covar_module structure."""
        try:
            if hasattr(model, 'covar_module'):
                covar_module = model.covar_module
                module_type = type(covar_module).__name__
                
                # Check for specific kernel types
                if hasattr(covar_module, 'cat_kernel') or hasattr(covar_module, 'cont_kernel') or hasattr(covar_module, 'source_kernel'):
                    return 'CombinedKernel'
                elif hasattr(covar_module, 'base_kernel'):
                    return 'ScaleKernel'
                elif hasattr(covar_module, 'kernels'):
                    return 'MultiKernel'
                else:
                    return module_type
            return 'Unknown'
        except:
            return 'Unknown'
    
    def _save_parameters(self):
        """Save stored parameters to JSON file."""
        try:
            import json
            import os
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.save_file) if os.path.dirname(self.save_file) else '.', exist_ok=True)
            
            # Save parameters
            with open(self.save_file, 'w') as f:
                json.dump(self.stored_parameters, f, indent=2)
            
            if self.verbose:
                print(f"Final parameters saved to: {self.save_file}")
                
        except Exception as e:
            print(f"Error saving parameters to file: {e}")
    
    def get_stored_parameters(self):
        """Get the stored parameters for external use."""
        return self.stored_parameters.copy()
    
    def get_best_model_metrics(self):
        """
        Get the best model metrics for inclusion in results files.
        Returns metrics from the run with the lowest best_loss.
        
        If stored_parameters is empty (e.g., in multi-run training where callbacks are deep-copied),
        this will try to load from the saved JSON file.
        
        Returns:
            dict: Dictionary containing:
                - num_epochs: Number of epochs trained
                - best_epoch: Epoch when best loss was achieved
                - best_loss: Best loss value
                - jitter: Cholesky jitter value used
                - noise: Transformed noise parameter
                - outputscale: Transformed outputscale parameter
                - lengthscales: List of transformed lengthscale parameters
        """
        # If stored_parameters is empty, try to load from file
        if not self.stored_parameters:
            try:
                import json
                import os
                if os.path.exists(self.save_file):
                    with open(self.save_file, 'r') as f:
                        self.stored_parameters = json.load(f)
            except Exception:
                pass
        
        if not self.stored_parameters:
            return None
        
        # Find the run with the lowest best_loss
        best_run = min(
            [r for r in self.stored_parameters if r.get("best_loss") is not None],
            key=lambda x: x.get("best_loss", float("inf")),
            default=None
        )
        
        if best_run is None:
            return None
        
        # Extract relevant metrics
        final_dict = best_run.get("final", {})
        lengthscales = final_dict.get("lengthscales")
        
        # Debug output
        if self.verbose:
            print(f"[DEBUG get_best_model_metrics] best_run keys: {list(best_run.keys())}")
            print(f"[DEBUG get_best_model_metrics] final_dict keys: {list(final_dict.keys())}")
            if lengthscales is not None:
                if isinstance(lengthscales, (list, tuple)):
                    print(f"[DEBUG get_best_model_metrics] Found {len(lengthscales)} lengthscales in final dict")
                else:
                    print(f"[DEBUG get_best_model_metrics] lengthscales is not a list: {type(lengthscales)}, value: {lengthscales}")
            else:
                print(f"[DEBUG get_best_model_metrics] lengthscales is None in final_dict")
        
        metrics = {
            "num_epochs": best_run.get("num_epochs"),
            "best_epoch": best_run.get("best_epoch"),
            "best_loss": best_run.get("best_loss"),
            "jitter": best_run.get("jitter"),
            "noise": final_dict.get("noise"),
            "outputscale": final_dict.get("outputscale"),
            "lengthscales": lengthscales,
        }
        
        return metrics
    
    def clear_parameters(self):
        """Clear stored parameters."""
        self.stored_parameters = []