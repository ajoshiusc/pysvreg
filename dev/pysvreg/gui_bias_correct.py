"""Tk-based Python viewer and manual tissue-intensity correction utility."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from scipy.ndimage import gaussian_filter

from . import native


def _tissue_medians(image: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    result = {}
    for code in (1, 2, 3):
        values = np.asarray(image)[np.asarray(labels) == code]
        if values.size == 0:
            raise ValueError(f"tissue label {code} has no voxels")
        result[code] = float(np.median(values))
    return result


def tissue_median_correct(image: np.ndarray, labels: np.ndarray, targets: dict[int, float], sigma: float = 8.0) -> np.ndarray:
    """Apply a smooth log-gain field that matches selected tissue medians."""
    image = np.asarray(image, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    if image.shape != labels.shape:
        raise ValueError("image and tissue-label dimensions differ")
    current = _tissue_medians(image, labels)
    log_gain = np.zeros(image.shape, dtype=np.float32)
    tissue = np.zeros(image.shape, dtype=bool)
    for code in (1, 2, 3):
        target = float(targets[code])
        if not np.isfinite(target) or target <= 0 or current[code] <= 0:
            raise ValueError("target and current tissue medians must be finite and positive")
        mask = labels == code
        log_gain[mask] = np.log(target / current[code])
        tissue |= mask
    smooth_weight = gaussian_filter(tissue.astype(np.float32), sigma=sigma, mode="nearest")
    smooth_log = gaussian_filter(log_gain, sigma=sigma, mode="nearest")
    field = np.divide(smooth_log, smooth_weight, out=np.zeros_like(smooth_log), where=smooth_weight > 1e-5)
    return image * np.exp(field)


class BiasCorrectionWindow:
    def __init__(self, root: tk.Tk, image_path: Path):
        self.root = root
        self.image_path = image_path
        self.image = native.load_nii_z(str(image_path))
        self.original = np.asarray(self.image.img, dtype=np.float32)
        if self.original.ndim != 3:
            raise ValueError(f"expected a 3D NIfTI image, got {self.original.shape}")
        prefix = str(image_path)
        for suffix in (".nii.gz", ".nii"):
            if prefix.endswith(suffix):
                prefix = prefix[:-len(suffix)]
                break
        label_path = Path(prefix + ".pvc.label.nii.gz")
        if not label_path.is_file():
            label_path = Path(prefix + ".pvc.label.nii")
        if not label_path.is_file():
            raise FileNotFoundError(f"tissue labels are missing: {label_path}")
        self.labels = np.asarray(native.load_nii_z(str(label_path)).img, dtype=np.int32)
        if self.labels.shape != self.original.shape:
            raise ValueError(f"image and tissue-label dimensions differ: {self.original.shape} vs {self.labels.shape}")
        if not np.any(self.labels == 3):
            self.labels = self.labels + 1
        self.names = {1: "CSF", 2: "Gray matter", 3: "White matter"}
        self.medians = _tissue_medians(self.original, self.labels)

        self.corrected = self.original.copy()
        self.slice_index = tk.IntVar(value=self.original.shape[2] // 2)
        self.target_values = {code: tk.StringVar(value=f"{value:.2f}") for code, value in self.medians.items()}
        self.photo = None
        self._build()
        self._draw()

    def _build(self):
        self.root.title("SVReg manual bias correction")
        self.root.geometry("980x760")
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(main, width=760, height=650, background="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=6, sticky="nsew", padx=(0, 12))
        side = ttk.Frame(main)
        side.grid(row=0, column=1, sticky="nsew")
        ttk.Label(side, text=self.image_path.name, wraplength=200).pack(anchor="w", pady=(0, 12))
        ttk.Label(side, text="Tissue median targets").pack(anchor="w")
        for code in (1, 2, 3):
            line = ttk.Frame(side)
            line.pack(fill="x", pady=3)
            ttk.Label(line, text=f"{self.names[code]} (current {self.medians[code]:.2f})", width=24).pack(side="left")
            ttk.Entry(line, textvariable=self.target_values[code], width=9).pack(side="right")
        ttk.Label(
            side,
            text="The target medians define a smooth multiplicative field from the tissue labels. Set equal targets to keep the image unchanged.",
            wraplength=220,
            justify="left",
        ).pack(anchor="w", pady=10)
        ttk.Button(side, text="Preview correction", command=self._apply).pack(fill="x", pady=4)
        ttk.Button(side, text="Save corrected NIfTI…", command=self._save).pack(fill="x", pady=4)
        ttk.Button(side, text="Reset", command=self._reset).pack(fill="x", pady=4)
        self.slice_scale = ttk.Scale(
            side, from_=0, to=self.original.shape[2] - 1, orient="horizontal",
            command=self._slice_changed,
        )
        self.slice_scale.set(self.slice_index.get())
        self.slice_scale.pack(fill="x", pady=(18, 2))
        self.slice_label = ttk.Label(side, text="")
        self.slice_label.pack(anchor="w")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

    def _slice_changed(self, value):
        self.slice_index.set(int(float(value)))
        self._draw()

    def _draw(self):
        index = self.slice_index.get()
        self.slice_label.configure(text=f"Axial slice {index + 1} / {self.original.shape[2]}")
        slice_data = self.corrected[:, :, index]
        finite = slice_data[np.isfinite(slice_data)]
        if finite.size == 0:
            return
        lo, hi = np.percentile(finite, (1, 99))
        if hi <= lo:
            hi = lo + 1
        gray = np.clip((slice_data - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        ppm = f"P6\n{gray.shape[1]} {gray.shape[0]}\n255\n".encode("ascii") + rgb[::-1].transpose(1, 0, 2).tobytes()
        self.photo = tk.PhotoImage(data=base64.b64encode(ppm).decode("ascii"), format="PPM")
        self.canvas.delete("all")
        width, height = self.photo.width(), self.photo.height()
        scale = min(740 / width, 630 / height)
        display = self.photo.subsample(max(1, int(1 / scale))) if scale < 1 else self.photo
        self.canvas.create_image(380, 325, image=display, anchor="center")
        self.canvas.image = display

    def _apply(self):
        try:
            targets = {code: float(var.get()) for code, var in self.target_values.items()}
            self.corrected = tissue_median_correct(self.original, self.labels, targets)
            self._draw()
        except (ValueError, ZeroDivisionError) as exc:
            messagebox.showerror("Invalid tissue target", str(exc), parent=self.root)

    def _reset(self):
        self.corrected = self.original.copy()
        for code in (1, 2, 3):
            self.target_values[code].set(f"{self.medians[code]:.2f}")
        self._draw()

    def _save(self):
        destination = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save corrected NIfTI",
            initialdir=str(self.image_path.parent),
            initialfile=f"{self.image_path.name.removesuffix('.nii.gz').removesuffix('.nii')}_bias_corrected.nii.gz",
            defaultextension=".nii.gz",
            filetypes=(("Compressed NIfTI", "*.nii.gz"), ("NIfTI", "*.nii")),
        )
        if not destination:
            return
        header = self.image.header.copy()
        header.set_data_dtype(np.float32)
        native.save_untouch_nii_gz(
            native.NiftiVolume(self.corrected.astype(np.float32), self.image.affine, header, self.image.path),
            destination,
        )
        messagebox.showinfo("Saved", f"Corrected image saved to:\n{destination}", parent=self.root)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv:
            image_path = Path(argv[0]).expanduser().resolve()
        else:
            picker_root = tk.Tk()
            picker_root.withdraw()
            try:
                selected = filedialog.askopenfilename(
                    parent=picker_root,
                    title="Select image for bias correction",
                    filetypes=(("NIfTI images", "*.nii *.nii.gz"), ("All files", "*")),
                )
            finally:
                picker_root.destroy()
            if not selected:
                return 0
            image_path = Path(selected)
        root = tk.Tk()
        BiasCorrectionWindow(root, image_path)
        root.mainloop()
        return 0
    except Exception as exc:
        print(f"gui_bias_correct: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

