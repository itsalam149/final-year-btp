"""
src package for Task-Conditioned Evaluation of Quantized LLMs
Group ID: DTU/CSE/BTP/GP159
"""

from urllib.parse import unquote

# 1. Compatibility Patch: HfUriError & bare repository names across all HuggingFace datasets
try:
    import huggingface_hub.utils._hf_uris as hf_uris

    orig_post_init = hf_uris.HfUri.__post_init__

    def patched_post_init(self):
        uri = self._raw or ""
        if not self.id:
            raise hf_uris.HfUriError(uri=uri, msg="Missing repository id.")
        if self.revision is not None and not self.revision:
            raise hf_uris.HfUriError(uri=uri, msg="Revision must not be an empty string.")

    hf_uris.HfUri.__post_init__ = patched_post_init

    orig_parse_repo_body = hf_uris._parse_repo_body

    def robust_parse_repo_body(location: str, type_, *, raw: str = ""):
        location = location.strip("/")
        if not location:
            raise hf_uris.HfUriError(uri=raw, msg="Missing repository id.")

        at_idx = location.find("@")
        if at_idx == -1:
            parts = location.split("/", 1)
            if len(parts) == 1:
                repo_id = parts[0]
                path_in_repo = ""
            else:
                subparts = parts[1].split("/", 1)
                if len(subparts) == 1:
                    repo_id = f"{parts[0]}/{parts[1]}"
                    path_in_repo = ""
                else:
                    repo_id = f"{parts[0]}/{subparts[0]}"
                    path_in_repo = subparts[1]
            return hf_uris.HfUri(type=type_, id=repo_id, revision=None, path_in_repo=path_in_repo, _raw=raw)
        else:
            repo_id = location[:at_idx]
            rev_and_path = location[at_idx + 1:]
            slash_idx = rev_and_path.find("/")
            if slash_idx == -1:
                revision = rev_and_path
                path_in_repo = ""
            else:
                revision = rev_and_path[:slash_idx]
                path_in_repo = rev_and_path[slash_idx + 1:]
            return hf_uris.HfUri(type=type_, id=repo_id, revision=unquote(revision), path_in_repo=path_in_repo, _raw=raw)

    hf_uris._parse_repo_body = robust_parse_repo_body
except Exception:
    pass

# 2. Compatibility Patch: datasets.builder.DatasetBuilder data_files check
try:
    import datasets.builder as ds_builder
    orig_builder_init = ds_builder.DatasetBuilder.__init__

    def patched_builder_init(self, *args, **kwargs):
        try:
            orig_builder_init(self, *args, **kwargs)
        except ValueError as e:
            if "Data files don't belong to" in str(e):
                orig_repo_id = kwargs.get("repo_id")
                kwargs["repo_id"] = None
                orig_builder_init(self, *args, **kwargs)
                self.repo_id = orig_repo_id
            else:
                raise e

    ds_builder.DatasetBuilder.__init__ = patched_builder_init
except Exception:
    pass
