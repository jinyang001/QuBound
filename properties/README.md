# Backend property files

QuBound uses historical IBM backend property data, such as gate errors, readout errors, and coherence times, to construct noise-aware input features.

Large backend property JSON files are not tracked in this repository because they can exceed GitHub's normal file-size limits. To run the full dataset generation workflow, place the required files in this directory.

Expected filenames used by the current scripts include:

```text
ibmq_mumbai_properties.json
ibmq_kolkata_properties.json
```

Recommended options for sharing these files:

- GitHub Releases
- Google Drive
- Zenodo
- another external dataset hosting service
- Git LFS, if you specifically want to manage large files through GitHub

Do not commit IBM Quantum credentials, API tokens, local configuration files, or private account information.
