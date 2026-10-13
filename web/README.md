# AFM Web Lab · browser adapter v1.0

Live edition: [Chemitro AFM Web Lab](https://chemitro.ir/afm.php).
The [public software page](https://chemitro.ir/software.php?tool=afm-studio)
contains the desktop link, method overview, DOI and citation.

The browser edition runs the unchanged v0.4.0 scientific core in a dedicated
WebAssembly worker. Files remain in the worker's memory-only filesystem; the
application does not send scan contents to the website server.

## Included functionality

- Nanosurf NID import and numerical height matrices with explicit calibration.
- Forward/backward channel selection; processed, raw and difference views.
- Interactive 2D/3D maps, full-resolution line profiles and calibrated ROI crops.
- Huber/least-squares planes, row correction, line flattening and finite-aware filters.
- Full-raster roughness, physical PSD and linear overlap-normalized ACF.
- Original particle geometry, height/volume measurement and sampling/boundary QC.
- Local CSV, JSON and PNG downloads, with source fingerprints and processing history.

The browser supports files up to 32 MB, rasters up to 1024 × 1024 and eight undo
states. The desktop edition supplies batch analysis, Excel and 600 DPI figures.

## Hosting integration

`afm.php` uses Chemitro's shared PHP member authentication and theme. The public
discovery/citation page is available to guests; the lab follows the site's
existing EIS/XPS sign-in policy. `afm.css` extends the shared science/EIS styles.

The worker expects a same-origin Pyodide 314.0.7 runtime at
`assets/eis/runtime/`, including NumPy 2.4.6, SciPy 1.18.0 and scikit-image 0.26.0
and their official dependencies. Package files are verified against the runtime
lockfile. The PHP page limits connections and workers to the same origin and
permits WebAssembly compilation through `wasm-unsafe-eval`.

Copy the application files under `public_html/assets/afm/`, serving `bridge.py`
as `bridge.py.txt` and `source-manifest.json` as `source-manifest.txt`.
The manifest identifies five exact source modules from commit
`08454897c122f60b617d6909029fa94e9cb7ce2a`, the archived v0.4.0 tag.
Serve those modules at `assets/afm/source/<module-path>.txt`; their bytes and
SHA-256 values must match the manifest. The worker checks every module before import.

## Scientific conventions

Numerical analysis uses the complete raster; preview maps sample up to 192 × 192
pixels. Physical pixel pitch is field width divided by raster count. 3D vertical
proportions are amplified for visualization. PSD is a 2D-density annular mean;
the plotted ACF is its positive X section. Lateral particle dimensions remain
apparent AFM sizes. Sampling QC does not correct tip convolution or establish
formal uncertainty. Matrices use user-supplied calibration; NID badges compare
header fields. JSON reports describe analyses rather than restoring sessions.

## Validation

The web edition passed 34 scientific checks in the shipped WebAssembly runtime,
including binary import, source hashes, missing data, voltage-channel rejection,
crop geometry, known particle area/height/volume, and PSD Parseval normalization.
Another 20 checks cover PHP access, template integration, citation links and
content migration. Browser QA exercised the demo, leveling, particle/PSD analysis,
exact profile/crop controls, 3D view and mobile layout.

Cite the scientific core using [10.5281/zenodo.23348914](https://doi.org/10.5281/zenodo.23348914),
and record both core v0.4.0 and browser adapter v1.0 when reporting analyses.
