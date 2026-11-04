# Kaggle preparation and publication

Account: ndhoang2310. Credentials are read from the existing local Kaggle configuration and are never included in the notebook or dataset payload.

CPU notebook: https://www.kaggle.com/code/ndhoang2310/iir-b0-sfm100-cpu-preparation

Verified public dataset (version 1, ready): https://www.kaggle.com/datasets/ndhoang2310/iir-b0-sfm100-prepared

The CPU notebook is a copy of 01_prepare_sfm100_cpu.ipynb with only its exporter changed to stream the exact prepared layout into one ZIP_STORED archive. This avoids downloading 16,692 individual output files. The transport exporter was tested with a small fixture for byte equality, paths and manifest hashes. No credentials are sent to the remote notebook.

Scripts:

- status.py: queries job status and available output logs.
- publish_prepared.py download: requires a completed job; downloads result JSON and archive; verifies full archive SHA-256.
- publish_prepared.py publish: verifies the complete image inventory and every SHA-256 before creating the public dataset with license category Other, retained from the source dataset. Refuses unexpected upload files.
- verify_public.py: checks dataset readiness, anonymous visibility and extracted image/metadata layout.
- check_notebook_data.py: executes notebook 02's metadata assertions against the downloaded real subset; checks CID inventory, partition identity and the deployed subset/cleanup code against the saved baseline. Run with a Python environment containing NumPy.

Verified locally on 2026-10-29: the complete 5,844,749,328-byte archive SHA-256, all 16,692 image hashes, metadata hashes and notebook 02's data contract passed. Details are in validation-report.json and notebook-compatibility.json. The original Downloads notebook is no longer present, so comparison uses the saved baseline and notebook 02. GPU training was not run. Publication is confirmed only by public-verification.json.

Output images are original bytes, not resized tensors. Only the prepared SfM subset is uploaded; no Oxford/Paris images, weights, checkpoints or session logs are in the public payload. Runtime state and successful publication are recorded separately in kernel-status.json, preparation_result.json, validation-report.json, publish-result.json and public-verification.json as those steps complete. The existence of this document does not mean publication is complete.

Publication completed 2026-10-29: anonymous metadata access succeeded, is_private=false, and all 16,696 extracted file paths (16,692 images plus four metadata/manifest files) match the verified archive. The prepared manifest is at the dataset root; notebook 02 can discover it automatically. Keep the benchmark and Caffe weights datasets attached separately.
