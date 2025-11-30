import importlib.metadata

def check_package(name):
    try:
        version = importlib.metadata.version(name)
        print(f"✅ {name}: {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"❌ {name}: NOT INSTALLED")

print("🔍 Checking Library Versions...")
check_package("youtube-transcript-api")  # Should be >= 0.6.0
check_package("qdrant-client")           # Should be >= 1.7.0
