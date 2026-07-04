"""EngagementDataset — the loader contract (all-nodes session windows).

Each __getitem__ returns one width-W window of a session with **every** node in it: per-modality
streams, presence/validity, the scored-target mask, and per-node ground-truth chunks Y_t. The model
runs the per-node trunk once for all nodes and then a target-specific graph + head per node, so one
forward predicts engagement for everyone in the frame (and training loops over all people too).
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from . import registry, streams, windowing
from .normalize import Normalizer, empty as empty_normalizer
from .windowing import resample_categorical_nearest
from .sessions import SampleRecord


def _read_float_csv(path: str) -> np.ndarray:
    vals = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(float(s))
            except ValueError:
                vals.append(np.nan)
    return np.asarray(vals, dtype=np.float32)


def _count_lines(path: str) -> int:
    """Non-empty line count (== label frame count) without parsing values."""
    if not path or not os.path.exists(path):
        return 0
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n


class EngagementDataset(Dataset):
    def __init__(self, records, config, normalizer: Normalizer | None = None,
                 train: bool = True, seed: int = 40):
        self.records: list[SampleRecord] = list(records)
        self.cfg = config
        self.W = int(config["window"]["W"])
        self.S = int(config["window"]["S"])
        self.K = int(config["window"]["K"])
        self.fps = registry.LABEL_FPS_CONTINUOUS
        self.train = train
        self.normalizer = normalizer if normalizer is not None else empty_normalizer()
        self._cache: OrderedDict = OrderedDict()
        self._cache_max = registry.MAX_PARTNERS + 1  # fits all roles of any active session (MAX_PARTNERS partners + target = 4)
        # Optional precomputed (normalized 25 fps fp16) feature dir. 
        self.precomputed_dir = (config.get("data", {}) or {}).get("precomputed_dir")
        self._T: list[int] = []
        self.windows: list[tuple[int, int]] = []
        for ri, rec in enumerate(self.records):
            T = self._record_T(rec)
            self._T.append(T)
            for s in windowing.window_starts(T, self.W, self.S):
                self.windows.append((ri, s))

    def __len__(self):
        return len(self.windows)

    # ── session length on the label grid (cheap line count) ───────────────────
    def _record_T(self, rec: SampleRecord) -> int:
        """Number of 25 fps frames for the session (max over scored target roles).

        Length comes from the annotation CSVs (line count). Test sessions ship no annotations, so
        when no label is found we fall back to the native frame count of the openpose stream — the
        role probe modality present for every node — converted to the 25 fps grid.
        """
        ref = rec.ref
        if ref.label_kind == "continuous":
            best = 1
            for role in rec.target_roles:
                path = (os.path.join(ref.label_dir,
                                     f"{role}.engagement.annotation.csv")
                        if ref.label_dir else "")
                best = max(best, _count_lines(path))
            return best if best > 1 else self._stream_T25(rec)
        best30 = 1
        for role in rec.target_roles:
            soc_p = os.path.join(ref.label_dir,
                                 f"{role}.social_engagement.annotation.csv")
            tsk_p = os.path.join(ref.label_dir,
                                 f"{role}.task_engagement.annotation.csv")
            best30 = max(best30, _count_lines(soc_p), _count_lines(tsk_p))
        if best30 > 1:
            return max(int(round(best30 * self.fps / registry.LABEL_FPS_PINSORO)), 1)
        return self._stream_T25(rec)

    def _stream_T25(self, rec: SampleRecord) -> int:
        """Native openpose frame count of the first available target role, on the 25 fps grid.

        Used when a session has no annotation CSV (test splits): ``T = round(num · 25 / sr)`` from
        the openpose ``.stream`` header (no labels needed).
        """
        ref = rec.ref
        suffix = f".{registry.ROLE_PROBE_MODALITY}.stream"
        for role in rec.target_roles:
            path = os.path.join(ref.feature_dir, f"{role}{suffix}")
            if streams.file_exists(path):
                num, sr, _ = streams.read_stream_header(path)
                if num > 0 and sr > 0:
                    return max(int(round(num * self.fps / sr)), 1)
        return 1

    # ── label loading (per role, on the 25 fps grid) ──────────────────────────
    def _load_role_labels(self, ref: registry.SessionRef, role: str, T: int):
        """Return the label payload for one role: ``{"continuous": (L,)}`` or
        ``{"social": (T,), "task": (T,)}`` (categorical, already resampled to 25 fps)."""
        if ref.label_kind == "continuous":
            path = os.path.join(ref.label_dir,
                                f"{role}.engagement.annotation.csv")
            lab = (_read_float_csv(path)
                   if ref.label_dir and os.path.exists(path)
                   else np.zeros(0, dtype=np.float32))
            return {"continuous": lab}
        soc_p = os.path.join(ref.label_dir,
                             f"{role}.social_engagement.annotation.csv")
        tsk_p = os.path.join(ref.label_dir,
                             f"{role}.task_engagement.annotation.csv")
        soc = (streams.read_str_csv_to_idx(soc_p, registry.SOCIAL_CLASSES)
               if os.path.exists(soc_p) else np.zeros(0, dtype=np.int64))
        tsk = (streams.read_str_csv_to_idx(tsk_p, registry.TASK_CLASSES)
               if os.path.exists(tsk_p) else np.zeros(0, dtype=np.int64))
        soc = resample_categorical_nearest(soc, T) if len(soc) else np.full(T, -1, np.int64)
        tsk = resample_categorical_nearest(tsk, T) if len(tsk) else np.full(T, -1, np.int64)
        return {"social": soc, "task": tsk}

    def _load_node(self, ref: registry.SessionRef, role: str, T: int):
        """Return ({modality: (T, D_m)}, present[M]) for one node, cached.

        Loads from the precomputed memmap dir when configured (both continuous and categorical);
        any missing/stale array falls back to parsing the raw stream for that modality.
        """
        key = (ref.feature_dir, ref.whisper_dir, role, T)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        if self.precomputed_dir:
            value = self._load_node_precomputed(ref, role, T)
        else:
            value = self._load_node_raw(ref, role, T)
        self._cache[key] = value
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return value

    def _load_modality_raw(self, ref: registry.SessionRef, role: str, modality: str, T: int):
        """Parse + normalize one raw modality stream -> ``((T, D_eff) float32, present: bool)``.

        ``apply()`` does channel selection, normalization, mapping invalid frames (NaN/inf/sentinel
        magnitudes, OpenPose -1) to the neutral post-norm value 0, and the ±10 clip. Missing streams
        return a zero array with ``present=False``.
        """
        pattern, kind = registry.MODALITY_FILE[modality]
        D_raw = registry.MODALITY_DIMS[modality]
        fname = pattern.format(role=role)
        if modality == "whisper":
            base = ref.whisper_dir
            path = os.path.join(base, fname) if base else ""
            ok = bool(base) and os.path.exists(path)
            arr = streams.load_whisper(path, T) if ok else None
        else:
            path = os.path.join(ref.feature_dir, fname)
            ok = streams.file_exists(path)
            arr = (streams.load_stream(path, T, T / self.fps) if ok else None)
        present = arr is not None and arr.shape[1] == D_raw
        if not present:
            arr = np.zeros((T, D_raw), dtype=np.float32)
        return self.normalizer.apply(modality, arr), bool(present)

    def _load_node_raw(self, ref: registry.SessionRef, role: str, T: int):
        """Raw-stream node load: ({modality: (T, D_eff)}, present[M]) for one node."""
        feats = {}
        present = np.zeros(len(registry.MODALITY_ORDER), dtype=np.float32)
        for mi, modality in enumerate(registry.MODALITY_ORDER):
            arr, ok = self._load_modality_raw(ref, role, modality, T)
            feats[modality] = arr
            if ok:
                present[mi] = 1.0
        return feats, present

    def _precomputed_session_dir(self, ref: registry.SessionRef) -> str:
        """Precompute dir for a session, mirroring its feature_dir relative to the dataset root."""
        root = self.cfg["data"]["roots"][ref.dataset]
        rel = os.path.relpath(ref.feature_dir, root)
        return os.path.join(self.precomputed_dir, ref.dataset, rel)

    def _load_node_precomputed(self, ref: registry.SessionRef, role: str, T: int):
        """Memmap node load from the precomputed dir: ({modality: (T, D_eff)}, present[M]).

        Present modalities are memory-mapped fp16 arrays; absent
        modalities are a zero-broadcast view (the model masks them out via ``node_present``). Falls
        back to the raw loader when the session/role was not precomputed
        """
        pre_dir = self._precomputed_session_dir(ref)
        eff_dims = self.normalizer.effective_dims
        role_has_any = os.path.isdir(pre_dir) and any(
            os.path.exists(os.path.join(pre_dir, f"{role}.{m}.npy"))
            for m in registry.MODALITY_ORDER)
        if not role_has_any:
            return self._load_node_raw(ref, role, T)     # not precomputed -> raw fallback

        feats = {}
        present = np.zeros(len(registry.MODALITY_ORDER), dtype=np.float32)
        for mi, modality in enumerate(registry.MODALITY_ORDER):
            D_eff = eff_dims[modality]
            path = os.path.join(pre_dir, f"{role}.{modality}.npy")
            if os.path.exists(path):
                arr = np.load(path, mmap_mode="r")
                if arr.shape == (T, D_eff):
                    feats[modality] = arr                  # memmap; sliced in _slice_node
                    present[mi] = 1.0
                    continue
                # stale precompute (T or dim changed) -> recompute this one modality from raw.
                arr, ok = self._load_modality_raw(ref, role, modality, T)
                feats[modality] = arr
                if ok:
                    present[mi] = 1.0
                continue
            # absent modality: zero-broadcast view (O(D) memory; masked out by node_present).
            feats[modality] = np.broadcast_to(
                np.zeros((1, D_eff), dtype=np.float16), (T, D_eff))
        return feats, present

    # ── windowing ────────────────────────────────────────────────────────────
    def _slice_node(self, feats: dict, present: np.ndarray, start: int):
        """Window each *present* modality to ``(W, D)``, preserving the source dtype.

        Absent modalities are omitted (collate's zero slot stands in). The window keeps the array's
        own dtype — fp16 for precomputed features, fp32 for raw streams.
        """
        out = {}
        for mi, modality in enumerate(registry.MODALITY_ORDER):
            if present[mi] < 0.5:
                continue                                  # absent: collate's zero slot stands in
            arr = feats[modality]
            T, D = arr.shape
            win = np.zeros((self.W, D), dtype=arr.dtype)
            n = max(0, min(self.W, T - start))
            win[:n] = arr[start:start + n]
            out[modality] = torch.from_numpy(win)
        return out

    def __getitem__(self, i: int):
        rec_idx, start = self.windows[i]
        rec = self.records[rec_idx]
        ref = rec.ref
        T = self._T[rec_idx]
        node_roles = list(rec.node_roles)
        target_set = set(rec.target_roles)
        n = len(node_roles)

        nodes, node_present, node_fv, is_target, role_idx, framing_idx = [], [], [], [], [], []
        for role in node_roles:
            feats, present = self._load_node(ref, role, T)
            nodes.append(self._slice_node(feats, present, start))
            node_present.append(torch.from_numpy(present))
            node_fv.append(windowing.frames_valid(T, start, self.W))
            is_target.append(1.0 if role in target_set else 0.0)
            rc = registry.role_class_id(ref.dataset, role, ref.feature_dir)
            role_idx.append(rc)
            framing_idx.append(registry.framing_for_role_class(rc))

        sample = {
            "dataset_id": ref.dataset,
            "domain": rec.domain,
            "session_id": ref.session_id,
            "feature_dir": ref.feature_dir,
            "node_roles": node_roles,
            "label_kind": ref.label_kind,
            "window_start": start,                 # for temporal ensembling
            "nodes": nodes,                        # list[n] of {m: (W, D_m)} — present modalities only
            "node_present": node_present,          # list[n] of (M,)
            "node_frames_valid": node_fv,          # list[n] of (W,)
            "is_target": is_target,                # list[n] of float (scored?)
            # ── domain-adaptation conditioning ids (DomainFiLM) ──
            "role_idx": role_idx,                  # list[n] of semantic role id
            "framing_idx": framing_idx,            # list[n] of per-node spatial-framing id
            "partner_count_idx": min(max(n - 1, 0), registry.MAX_PARTNERS),
            "label_kind_idx": registry.label_kind_id(ref.label_kind),
            "language_idx": registry.language_id(ref.feature_dir, ref.dataset),
            # NB: the active-sensor "modality config" rides on node_present (per-node multi-hot).
        }

        # Per-node ground-truth chunk Y_t = labels[start : start+K] (invalid for non-targets).
        chunk_start = start + self.W - self.K   # context = first (W-K) frames; predict the rest
        if ref.label_kind == "continuous":
            chunks, masks = [], []
            for role in node_roles:
                if role in target_set:
                    lab = self._load_role_labels(ref, role, T)["continuous"]
                    ch, mk = windowing.chunk_continuous(lab, chunk_start, self.K)
                else:
                    ch = torch.zeros(self.K, dtype=torch.float32)
                    mk = torch.zeros(self.K, dtype=torch.float32)
                chunks.append(ch)
                masks.append(mk)
            sample["target_chunk"] = torch.stack(chunks)        # (n, K)
            sample["valid_mask"] = torch.stack(masks)           # (n, K)
        else:
            sc, sm, tc, tm = [], [], [], []
            for role in node_roles:
                if role in target_set:
                    lab = self._load_role_labels(ref, role, T)
                    s_c, s_m = windowing.chunk_categorical(lab["social"], chunk_start, self.K)
                    t_c, t_m = windowing.chunk_categorical(lab["task"], chunk_start, self.K)
                else:
                    s_c = torch.full((self.K,), -1, dtype=torch.long)
                    t_c = torch.full((self.K,), -1, dtype=torch.long)
                    s_m = torch.zeros(self.K, dtype=torch.float32)
                    t_m = torch.zeros(self.K, dtype=torch.float32)
                sc.append(s_c); sm.append(s_m); tc.append(t_c); tm.append(t_m)
            sample["target_chunk"] = {"social": torch.stack(sc), "task": torch.stack(tc)}
            sample["valid_mask"] = {"social": torch.stack(sm), "task": torch.stack(tm)}
        return sample
