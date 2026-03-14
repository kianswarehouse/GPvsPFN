from tabpfn import TabPFNRegressor
import numpy as np

print("="*60)
print("Checking TabPFN Version")
print("="*60)

# Create model and fit it
model = TabPFNRegressor(device='cpu')
print(f"Model class: {type(model).__name__}")
print(f"Module: {model.__module__}")

# Fit on dummy data to initialize
X = np.random.randn(20, 5)
y = np.random.randn(20)
print("\nFitting model on dummy data...")
model.fit(X, y)

# Check all possible locations for config
print("\nSearching for config...")

# Method 1: Check inference_engine
if hasattr(model, 'inference_engine_'):
    engine = model.inference_engine_
    print(f"✓ Found inference_engine_")
    if hasattr(engine, 'model'):
        print(f"  ✓ Found engine.model")
        if hasattr(engine.model, 'config'):
            config = engine.model.config
            nlayers = getattr(config, 'nlayers', None)
            if nlayers:
                print(f"\n{'='*60}")
                print(f"RESULT: Number of layers = {nlayers}")
                if nlayers == 12:
                    print(f"✓ You are using TabPFN MODEL v2.0 (12 layers)")
                elif nlayers == 18:
                    print(f"✓ You are using TabPFN MODEL v2.5 (18 layers)")
                else:
                    print(f"? Unknown version (expected 12 or 18 layers)")
                print(f"{'='*60}")
            else:
                print(f"  ✗ config.nlayers not found")
                print(f"  Available config attrs: {dir(config)[:10]}")

# Method 2: Direct model access
if hasattr(model, 'model_'):
    print(f"✓ Found model_")
    if hasattr(model.model_, 'nlayers'):
        print(f"  Number of layers: {model.model_.nlayers}")

# Method 3: Check package version
try:
    import tabpfn
    if hasattr(tabpfn, '__version__'):
        print(f"\nPackage version: {tabpfn.__version__}")
except:
    pass

print("\n" + "="*60)
# If 12 = you're using model v2.0
# If 18 = you're using model v2.5