import importlib

# Libraries and their expected versions
libs = {
    "pandas": "2.2.2",
    "numpy": "1.26.4",
    "scikit-learn": "1.6.1",
    "lightgbm": "4.6.0",
    "torch": "2.9.0+cpu",
    "transformers": "4.57.1",
    "optuna": "4.2.1",
    "flask": "3.1.0"
}

for lib, expected in libs.items():
    try:
        module = importlib.import_module(lib)
        version = getattr(module, "__version__", "unknown")
        status = "✅ OK" if version == expected else f"⚠️ Mismatch (expected {expected})"
        print(f"{lib:<15} Installed: {version:<15} {status}")
    except ImportError:
        print(f"{lib:<15} ❌ Not installed")
