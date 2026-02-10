UMAP Example - Linux x64 (Debian/Ubuntu)
========================================

REQUIREMENTS:
------------
- Linux x64 (Debian, Ubuntu, or compatible)
- .NET 8.0 Runtime installed
  - Install: sudo apt install dotnet-runtime-8.0
  - Or: wget https://dot.net/v1/dotnet-install.sh && chmod +x dotnet-install.sh && ./dotnet-install.sh --channel 8.0

QUICK START:
-----------
1. Upload entire folder to Linux server
2. Make executable: chmod +x UMAPuwotSharp.Example
3. Run: ./UMAPuwotSharp.Example

Or with dotnet: dotnet UMAPuwotSharp.Example.dll

FILES TO UPLOAD:
---------------
✓ libuwot.so                    - Native library (795KB)
✓ UMAPuwotSharp.dll             - Main library (29KB)
✓ UMAPuwotSharp.Example         - Executable script (71KB)
✓ UMAPuwotSharp.Example.dll     - Example code (28KB)
✓ UMAPuwotSharp.Example.deps.json
✓ UMAPuwotSharp.Example.runtimeconfig.json

EXPECTED RUNTIME: 2-5 minutes

WHAT THE EXAMPLE DOES:
---------------------
- Demo 3: Model save/load with 500 samples
- Demo 1: 10,000 samples → 27D embedding
- Demo 2: Multi-dimensional test (1D to 50D)
- Demo 4: All 5 distance metrics
- Demo 5: Safety analysis with outlier detection
- Demo 6: Smart spread parameter testing
- Demo 7: 10k sample transform

TEMPORARY FILES CREATED:
-----------------------
- demo_model.umap (auto-deleted after demo)
- mammoth_10k_hnsw.umap (can be deleted)

TROUBLESHOOTING:
---------------
"Cannot open libuwot.so": Ensure libuwot.so is in same directory
"Permission denied": Run chmod +x UMAPuwotSharp.Example
"dotnet not found": Install .NET 8.0 runtime

VERSION: 3.42.2
