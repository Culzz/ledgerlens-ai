# data

Local data. Nothing in here is committed except this file.

```
data/
  finqa/     FinQA json (train.json, dev.json, test.json)
  uploads/   user uploads, private storage
```

## FinQA

```
python scripts/download_data.py
```

The script records the source URL and a checksum so runs are reproducible. Splits are used the way
they're published and never reshuffled. More on that in `docs/evaluation_protocol.md`.

## Uploads

`data/uploads/` is private local storage. Uploaded filenames are never used as paths, the API
generates its own names and keeps the original only as metadata.

Both directories are git-ignored. Don't commit datasets, uploads, or anyone's financial documents.
