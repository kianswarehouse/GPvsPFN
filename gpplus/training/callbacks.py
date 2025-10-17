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
    
    def on_train_start(self, context: dict):
        """Capture initial parameters at the start of training."""
        model = context["model"]
        self._run_count += 1
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
    
    def on_train_end(self, context: dict):
        """Store final parameters at the end of training."""
        model = context["model"]
        epoch = context["epoch"]
        best_loss = context.get("best_loss", None)
        
        try:
            # Extract parameters
            final_params = self._extract_final_parameters(model, epoch, best_loss)
            record = self._combine_initial_final(self._initial_params, final_params)
            self.stored_parameters.append(record)
            
            if self.verbose:
                print(f"\n=== Final Parameters (Run {self._run_count}, Epoch {epoch}) ===")
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
            "best_loss": final.get("best_loss"),
            "timestamp": final.get("timestamp"),
            "initial": None,
            "final": None,
            "deltas": None,
        }

        record["final"] = {
            "raw_noise": final.get("raw_noise"),
            "raw_outputscale": final.get("raw_outputscale"),
            "raw_lengthscales": final.get("raw_lengthscales"),
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
    
    def _extract_final_parameters(self, model, epoch: int, best_loss: float = None):
        """Extract raw parameters from the model using recursive search."""
        params = {
            "epoch": epoch,
            "best_loss": best_loss,
            "timestamp": None,
            "raw_noise": None,
            "raw_outputscale": None,
            "raw_lengthscales": None,
            "kernel_type": None,
            "input_dim": None
        }
        
        # Add timestamp
        from datetime import datetime
        params["timestamp"] = datetime.now().isoformat()
        
        # Recursively search for parameters
        noise_params = []
        outputscale_params = []
        lengthscale_params = []
        
        self._recursive_parameter_search(model, noise_params, outputscale_params, lengthscale_params)
        
        # Extract the first found parameter of each type
        if noise_params:
            params['raw_noise'] = noise_params[0]
        
        if outputscale_params:
            params['raw_outputscale'] = outputscale_params[0]
        
        if lengthscale_params:
            params['raw_lengthscales'] = lengthscale_params
        
        # Try to determine kernel type and input dimension
        params['kernel_type'] = self._determine_kernel_type(model)
        if hasattr(model, 'train_inputs') and model.train_inputs:
            params['input_dim'] = model.train_inputs[0].shape[-1]
        
        return params
    
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
    
    def clear_parameters(self):
        """Clear stored parameters."""
        self.stored_parameters = []