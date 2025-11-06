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
                print(f"Raw lengthscales: {self._initial_params['raw_lengthscales']}")
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
                print(f"Lengthscales (transformed): {final_params.get('lengthscales', 'N/A')}")
                print(f"Raw noise: {final_params['raw_noise']}")
                print(f"Raw outputscale: {final_params['raw_outputscale']}")
                print(f"Raw lengthscales: {final_params['raw_lengthscales']}")
                print(f"Best loss: {best_loss}")
                # If we have initial, also print deltas
                if self._initial_params is not None:
                    deltas = record.get('deltas', {})
                    print("--- Deltas (final - initial) ---")
                    print(f"Delta raw_noise: {deltas.get('raw_noise')}")
                    print(f"Delta raw_outputscale: {deltas.get('raw_outputscale')}")
                    print(f"Delta raw_lengthscales: {deltas.get('raw_lengthscales')}")
            
            # Save to file
            self._save_parameters()
            
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
            "noise": final.get("noise"),  # Transformed
            "outputscale": final.get("outputscale"),  # Transformed
            "lengthscales": final.get("lengthscales"),  # Transformed
            "kernel_type": final.get("kernel_type"),
            "input_dim": final.get("input_dim"),
        }

        if initial is None:
            return record

        record["initial"] = {
            "raw_noise": initial.get("raw_noise"),
            "raw_outputscale": initial.get("raw_outputscale"),
            "raw_lengthscales": initial.get("raw_lengthscales"),
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
        
        if lengthscale_params:
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
            if hasattr(obj, 'lengthscale'):
                try:
                    lengthscale = obj.lengthscale
                    if lengthscale.numel() > 0:
                        lengthscale_list = lengthscale.detach().cpu().numpy().flatten().tolist()
                        # Only add if we haven't found lengthscales yet, or merge if multiple
                        if not params["lengthscales"]:
                            params["lengthscales"] = lengthscale_list
                        else:
                            # Merge lengthscales (for multi-kernel scenarios)
                            params["lengthscales"].extend(lengthscale_list)
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
        metrics = {
            "num_epochs": best_run.get("num_epochs"),
            "best_epoch": best_run.get("best_epoch"),
            "best_loss": best_run.get("best_loss"),
            "jitter": best_run.get("jitter"),
            "noise": best_run.get("final", {}).get("noise"),
            "outputscale": best_run.get("final", {}).get("outputscale"),
            "lengthscales": best_run.get("final", {}).get("lengthscales"),
        }
        
        return metrics
    
    def clear_parameters(self):
        """Clear stored parameters."""
        self.stored_parameters = []