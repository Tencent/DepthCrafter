import sys
import os
import csv
import time
import subprocess
import threading

from pathlib import Path

import cv2
import psutil

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QTextEdit,
    QSpinBox,
    QDoubleSpinBox
)

from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


# ================================================================
# BASE DIRECTORY
# ================================================================

base_dir = Path(__file__).parent


# ================================================================
# SYSTEM MONITOR
# ================================================================

class SystemMonitor:
    """
    Monitor CPU, RAM, GPU and VRAM during one inference.

    Important design:

        START MONITOR
             |
             v
        Collect samples
             |
             v
        Process finishes
             |
             v
        Capture finish timestamp
             |
             v
        STOP MONITOR
             |
             v
        Use only samples <= finish timestamp

    We NEVER perform a new GPU sample after the process finishes.
    """

    def __init__(self, interval=0.1):

        self.interval = interval

        self.stop_event = threading.Event()

        self.thread = None

        self.lock = threading.Lock()

        self.samples = []

        self.start_time = None
        self.finish_time = None

    # ============================================================
    # START
    # ============================================================

    def start(self):

        with self.lock:
            self.samples = []

        self.stop_event.clear()

        self.start_time = time.perf_counter()

        self.finish_time = None

        self.thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )

        self.thread.start()

    # ============================================================
    # MONITOR LOOP
    # ============================================================

    def _monitor_loop(self):

        # Prime CPU measurement
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

        while not self.stop_event.is_set():

            # ----------------------------------------------------
            # Collect the complete sample first.
            # ----------------------------------------------------

            sample = self._collect_sample()

            # ----------------------------------------------------
            # Timestamp AFTER collecting the sample.
            #
            # This is important because nvidia-smi itself may
            # take some time.
            # ----------------------------------------------------

            sample_time = time.perf_counter()

            with self.lock:

                self.samples.append(
                    (
                        sample_time,
                        sample
                    )
                )

            # ----------------------------------------------------
            # Wait before next sample.
            # ----------------------------------------------------

            self.stop_event.wait(
                self.interval
            )

    # ============================================================
    # COLLECT SAMPLE
    # ============================================================

    def _collect_sample(self):

        # --------------------------------------------------------
        # CPU
        # --------------------------------------------------------

        try:

            cpu_percent = psutil.cpu_percent(
                interval=None
            )

        except Exception:

            cpu_percent = 0.0

        # --------------------------------------------------------
        # RAM
        # --------------------------------------------------------

        try:

            memory = psutil.virtual_memory()

            ram_gb = (
                memory.used /
                (1024 ** 3)
            )

            ram_percent = memory.percent

        except Exception:

            ram_gb = 0.0
            ram_percent = 0.0

        # --------------------------------------------------------
        # GPU
        # --------------------------------------------------------

        gpu_util = 0.0
        gpu_memory = 0.0
        gpu_memory_total = 0.0

        try:

            result = subprocess.run(
                [
                    "nvidia-smi",

                    "--query-gpu="
                    "index,"
                    "utilization.gpu,"
                    "memory.used,"
                    "memory.total",

                    "--format="
                    "csv,noheader,nounits"
                ],

                capture_output=True,

                text=True,

                shell=False,

                timeout=0.5
            )

            if result.returncode == 0:

                lines = (
                    result.stdout
                    .strip()
                    .splitlines()
                )

                for line in lines:

                    values = [
                        value.strip()
                        for value in line.split(",")
                    ]

                    if len(values) < 4:
                        continue

                    try:

                        current_gpu = float(
                            values[1]
                        )

                        current_memory = float(
                            values[2]
                        )

                        current_total = float(
                            values[3]
                        )

                    except (
                        ValueError,
                        TypeError
                    ):

                        continue

                    # ------------------------------------------------
                    # For multiple GPUs:
                    #
                    # GPU utilization = highest GPU utilization
                    # VRAM = highest memory usage
                    # ------------------------------------------------

                    gpu_util = max(
                        gpu_util,
                        current_gpu
                    )

                    if current_memory > gpu_memory:

                        gpu_memory = current_memory

                        gpu_memory_total = (
                            current_total
                        )

        except Exception:

            pass

        return {

            "gpu_util":
                gpu_util,

            "gpu_memory":
                gpu_memory,

            "gpu_memory_total":
                gpu_memory_total,

            "cpu":
                cpu_percent,

            "ram_gb":
                ram_gb,

            "ram_percent":
                ram_percent
        }

    # ============================================================
    # STOP
    # ============================================================

    def stop(self, finish_time):

        self.finish_time = finish_time

        # --------------------------------------------------------
        # Tell monitor thread to stop.
        # --------------------------------------------------------

        self.stop_event.set()

        # --------------------------------------------------------
        # Wait for the current monitoring operation.
        #
        # We DO NOT perform another sample here.
        # --------------------------------------------------------

        if self.thread is not None:

            self.thread.join(
                timeout=1.0
            )

        # --------------------------------------------------------
        # Only use samples that completed BEFORE or exactly at
        # process finish.
        # --------------------------------------------------------

        with self.lock:

            valid_samples = [

                sample

                for timestamp, sample
                in self.samples

                if timestamp <= finish_time
            ]

        return self.calculate_statistics(
            valid_samples
        )

    # ============================================================
    # CALCULATE STATISTICS
    # ============================================================

    def calculate_statistics(
        self,
        samples
    ):

        if not samples:

            return {

                "peak_gpu": 0.0,
                "avg_gpu": 0.0,

                "peak_vram": 0.0,
                "avg_vram": 0.0,
                "vram_total": 0.0,

                "peak_cpu": 0.0,
                "avg_cpu": 0.0,

                "peak_ram": 0.0,
                "avg_ram": 0.0,

                "peak_ram_percent": 0.0,
                "avg_ram_percent": 0.0,

                "sample_count": 0
            }

        # --------------------------------------------------------
        # GPU
        # --------------------------------------------------------

        gpu_values = [

            sample["gpu_util"]

            for sample in samples
        ]

        # --------------------------------------------------------
        # VRAM
        # --------------------------------------------------------

        vram_values = [

            sample["gpu_memory"]

            for sample in samples
        ]

        # --------------------------------------------------------
        # CPU
        # --------------------------------------------------------

        cpu_values = [

            sample["cpu"]

            for sample in samples
        ]

        # --------------------------------------------------------
        # RAM
        # --------------------------------------------------------

        ram_values = [

            sample["ram_gb"]

            for sample in samples
        ]

        ram_percent_values = [

            sample["ram_percent"]

            for sample in samples
        ]

        # --------------------------------------------------------
        # VRAM TOTAL
        # --------------------------------------------------------

        vram_totals = [

            sample["gpu_memory_total"]

            for sample in samples

            if sample["gpu_memory_total"] > 0
        ]

        if vram_totals:

            vram_total = max(
                vram_totals
            )

        else:

            vram_total = 0.0

        # --------------------------------------------------------
        # Return statistics
        # --------------------------------------------------------

        return {

            # GPU
            "peak_gpu":
                max(gpu_values),

            "avg_gpu":
                sum(gpu_values)
                / len(gpu_values),

            # VRAM
            "peak_vram":
                max(vram_values),

            "avg_vram":
                sum(vram_values)
                / len(vram_values),

            "vram_total":
                vram_total,

            # CPU
            "peak_cpu":
                max(cpu_values),

            "avg_cpu":
                sum(cpu_values)
                / len(cpu_values),

            # RAM
            "peak_ram":
                max(ram_values),

            "avg_ram":
                sum(ram_values)
                / len(ram_values),

            "peak_ram_percent":
                max(ram_percent_values),

            "avg_ram_percent":
                sum(ram_percent_values)
                / len(ram_percent_values),

            # Samples
            "sample_count":
                len(samples)
        }


# ================================================================
# DEPTH CRAFTER
# ================================================================

class DepthCrafter(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "DepthCrafter"
        )

        self.setGeometry(
            100,
            100,
            700,
            500
        )

        # ========================================================
        # Output area
        # ========================================================

        self.output_area = QTextEdit(
            self
        )

        self.output_area.setReadOnly(
            True
        )

        # ========================================================
        # Button
        # ========================================================

        self.button = QPushButton(
            "Run",
            self
        )

        self.button.clicked.connect(
            self.run_script
        )

        # ========================================================
        # Input / Output
        # ========================================================

        self.path_label = QLabel(
            "Select video path:"
        )

        self.path_field = QLineEdit()

        self.output_label = QLabel(
            "Select output path:"
        )

        self.output_field = QLineEdit()

        self.browse_button = QPushButton(
            "Browse"
        )

        self.out_browse_button = QPushButton(
            "Browse"
        )

        self.browse_button.clicked.connect(
            self.browse_file
        )

        self.out_browse_button.clicked.connect(
            self.browse_output_path
        )

        path_layout = QHBoxLayout()

        path_layout.addWidget(
            self.path_label
        )

        path_layout.addWidget(
            self.path_field
        )

        path_layout.addWidget(
            self.browse_button
        )

        out_path_layout = QHBoxLayout()

        out_path_layout.addWidget(
            self.output_label
        )

        out_path_layout.addWidget(
            self.output_field
        )

        out_path_layout.addWidget(
            self.out_browse_button
        )

        # ========================================================
        # Parameters
        # ========================================================

        self.seed_label = QLabel(
            "Seed"
        )

        self.seed_field = QSpinBox()

        self.seed_field.setRange(
            0,
            999999
        )

        self.seed_field.setValue(
            42
        )

        # --------------------------------------------------------

        self.guidance_label = QLabel(
            "Guidance Scale"
        )

        self.guidance_field = QDoubleSpinBox()

        self.guidance_field.setDecimals(
            2
        )

        self.guidance_field.setRange(
            0.0,
            20.0
        )

        self.guidance_field.setSingleStep(
            0.1
        )

        self.guidance_field.setValue(
            1.0
        )

        # --------------------------------------------------------

        self.window_label = QLabel(
            "Window Size"
        )

        self.window_field = QSpinBox()

        self.window_field.setRange(
            1,
            500
        )

        self.window_field.setValue(
            110
        )

        # --------------------------------------------------------

        self.overlap_label = QLabel(
            "Overlap"
        )

        self.overlap_field = QSpinBox()

        self.overlap_field.setRange(
            0,
            500
        )

        self.overlap_field.setValue(
            25
        )

        # --------------------------------------------------------

        self.maxres_label = QLabel(
            "Max Resolution"
        )

        self.maxres_field = QSpinBox()

        self.maxres_field.setRange(
            128,
            4096
        )

        self.maxres_field.setSingleStep(
            64
        )

        self.maxres_field.setValue(
            1024
        )

        # --------------------------------------------------------

        self.inference_label = QLabel(
            "Inference Steps"
        )

        self.inference_field = QSpinBox()

        self.inference_field.setRange(
            1,
            100
        )

        self.inference_field.setValue(
            5
        )

        # ========================================================
        # Parameter layout
        # ========================================================

        param_layout = QGridLayout()

        param_layout.addWidget(
            self.seed_label,
            0,
            0
        )

        param_layout.addWidget(
            self.seed_field,
            0,
            1
        )

        param_layout.addWidget(
            self.guidance_label,
            0,
            2
        )

        param_layout.addWidget(
            self.guidance_field,
            0,
            3
        )

        param_layout.addWidget(
            self.window_label,
            1,
            0
        )

        param_layout.addWidget(
            self.window_field,
            1,
            1
        )

        param_layout.addWidget(
            self.overlap_label,
            1,
            2
        )

        param_layout.addWidget(
            self.overlap_field,
            1,
            3
        )

        param_layout.addWidget(
            self.maxres_label,
            2,
            0
        )

        param_layout.addWidget(
            self.maxres_field,
            2,
            1
        )

        param_layout.addWidget(
            self.inference_label,
            2,
            2
        )

        param_layout.addWidget(
            self.inference_field,
            2,
            3
        )

        # ========================================================
        # Main layout
        # ========================================================

        # Add logo at the top
        self.logo_label = QLabel(self)
        try:
            pixmap = QPixmap(os.path.join(base_dir, "logo.png"))
            if not pixmap.isNull():
                # Scale the logo to fit nicely in the UI (adjust as needed)
                scaled_pixmap = pixmap.scaled(200, 100, aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
                self.logo_label.setPixmap(scaled_pixmap)
            else:
                # If logo fails to load, show a placeholder text
                self.logo_label.setText("Logo Not Found")
        except Exception as e:
            # If logo fails to load, show a placeholder text
            self.logo_label.setText("Logo Not Found")
            print(f"Failed to load logo: {e}")

        self.logo_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()

        layout.addWidget(self.logo_label)
        layout.addLayout(
            path_layout
        )

        layout.addLayout(
            out_path_layout
        )

        layout.addLayout(
            param_layout
        )

        layout.addWidget(
            self.button
        )

        layout.addWidget(
            self.output_area
        )

        self.setLayout(
            layout
        )

    # =============================================================
    # START MONITORING
    # =============================================================

    def start_monitoring(self):

        self.monitor = SystemMonitor(
            interval=0.1
        )

        self.monitor.start()

    # =============================================================
    # STOP MONITORING
    # =============================================================

    def stop_monitoring(
        self,
        finish_time
    ):

        if not hasattr(
            self,
            "monitor"
        ):

            return self.empty_statistics()

        if self.monitor is None:

            return self.empty_statistics()

        stats = self.monitor.stop(
            finish_time
        )

        self.monitor = None

        return stats

    # =============================================================
    # EMPTY STATISTICS
    # =============================================================

    def empty_statistics(self):

        return {

            "peak_gpu": 0.0,
            "avg_gpu": 0.0,

            "peak_vram": 0.0,
            "avg_vram": 0.0,
            "vram_total": 0.0,

            "peak_cpu": 0.0,
            "avg_cpu": 0.0,

            "peak_ram": 0.0,
            "avg_ram": 0.0,

            "peak_ram_percent": 0.0,
            "avg_ram_percent": 0.0,

            "sample_count": 0
        }

    # =============================================================
    # RUN SCRIPT
    # =============================================================

    def run_script(self):

        video_dir = (
            self.path_field
            .text()
            .strip()
        )

        output_dir = (
            self.output_field
            .text()
            .strip()
        )

        # ---------------------------------------------------------
        # Validate input
        # ---------------------------------------------------------

        if not video_dir:

            QMessageBox.warning(
                self,
                "Input Required",
                "Please select the input video folder."
            )

            return

        if not output_dir:

            QMessageBox.warning(
                self,
                "Output Required",
                "Please select the output folder."
            )

            return

        if not os.path.isdir(
            video_dir
        ):

            QMessageBox.warning(
                self,
                "Invalid Input",
                "Input folder does not exist."
            )

            return

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        # ---------------------------------------------------------
        # Disable button while processing
        # ---------------------------------------------------------

        self.button.setEnabled(
            False
        )

        # ---------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------

        seed = (
            self.seed_field.value()
        )

        guidance_scale = (
            self.guidance_field.value()
        )

        window_size = (
            self.window_field.value()
        )

        overlap = (
            self.overlap_field.value()
        )

        max_res = (
            self.maxres_field.value()
        )

        # inference_steps = (
        #     self.inference_field.value()
        # )

        # ---------------------------------------------------------
        # Script
        # ---------------------------------------------------------

        script_path = os.path.join(
            base_dir,
            "run.py"
        )

        # ---------------------------------------------------------
        # CSV data
        # ---------------------------------------------------------

        csv_rows = []

        total_frames = 0

        total_runtime = 0.0

        # =========================================================
        # FIND VIDEOS
        # =========================================================

        video_extensions = (
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".mxf",
            ".webm"
        )

        video_files = []

        for root, dirs, files in os.walk(
            video_dir
        ):

            for file in sorted(files):

                if file.lower().endswith(
                    video_extensions
                ):

                    video_files.append(
                        os.path.join(
                            root,
                            file
                        )
                    )

        # ---------------------------------------------------------
        # Sort queue
        # ---------------------------------------------------------

        video_files.sort()

        if not video_files:

            self.button.setEnabled(
                True
            )

            QMessageBox.warning(
                self,
                "No Videos",
                "No supported video files were found."
            )

            return

        # =========================================================
        # UI
        # =========================================================

        self.output_area.clear()

        self.output_area.append(
            f"Found {len(video_files)} video(s).\n"
        )

        # =========================================================
        # PROCESS VIDEOS
        # =========================================================

        for index, file_path in enumerate(
            video_files,
            1
        ):

            self.output_area.append(
                "=" * 70
            )

            self.output_area.append(
                f"[{index}/{len(video_files)}] "
                f"Processing: {file_path}"
            )

            # -----------------------------------------------------
            # READ VIDEO INFORMATION
            # -----------------------------------------------------

            cap = cv2.VideoCapture(
                file_path
            )

            if not cap.isOpened():

                self.output_area.append(
                    "ERROR: Could not open video.\n"
                )

                continue

            frame_count = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            cap.release()

            # -----------------------------------------------------
            # Protect against invalid FPS
            # -----------------------------------------------------

            if fps <= 0:

                fps = 0.0

            actual_resolution = (
                f"{width}x{height}"
            )

            self.output_area.append(
                f"Input Resolution: "
                f"{actual_resolution}"
            )

            self.output_area.append(
                f"FPS: {fps:.2f}"
            )

            self.output_area.append(
                f"Frames: {frame_count}"
            )

            # =====================================================
            # COMMAND
            # =====================================================

            command = [

                sys.executable,

                script_path,

                "--video-path",
                file_path,

                "--save-folder",
                output_dir,

                "--target-fps",
                str(fps),

                "--process-length",
                str(frame_count),

                "--seed",
                str(seed),

                "--guidance-scale",
                str(guidance_scale),

                "--window-size",
                str(window_size),

                "--overlap",
                str(overlap),

                "--max-res",
                str(max_res),

                # "--inference-steps",
                # str(inference_steps)
            ]

            # =====================================================
            # START MONITOR
            #
            # IMPORTANT:
            # Start monitoring BEFORE subprocess.
            # =====================================================

            self.start_monitoring()

            # -----------------------------------------------------
            # Capture start timestamp
            # -----------------------------------------------------

            start_time = time.perf_counter()

            # Wall-clock timestamp used for identifying newly
            # generated output files.
            process_start_time = time.time()

            result = None

            # =====================================================
            # RUN PROCESS
            # =====================================================

            try:

                result = subprocess.run(
                    command,
                    shell=False
                )

            except Exception as e:

                self.output_area.append(
                    f"ERROR: {str(e)}"
                )

            # =====================================================
            # CAPTURE FINISH TIME FIRST
            #
            # Do this BEFORE stopping the monitor.
            # =====================================================

            finish_time = (
                time.perf_counter()
            )

            # =====================================================
            # EXACT RUNTIME
            # =====================================================

            runtime_seconds = (
                finish_time -
                start_time
            )

            runtime_string = (
                self.format_runtime(
                    runtime_seconds
                )
            )

            # =====================================================
            # STOP MONITORING
            #
            # No additional GPU sample is taken here.
            # =====================================================

            stats = self.stop_monitoring(
                finish_time
            )

            # =====================================================
            # PROCESSING FPS
            # =====================================================

            if runtime_seconds > 0:

                processing_fps = (
                    frame_count /
                    runtime_seconds
                )

            else:

                processing_fps = 0.0

            # =====================================================
            # OUTPUT RESOLUTION
            # =====================================================

            output_resolution = (
                self.get_output_resolution(
                    file_path,
                    output_dir,
                    process_start_time
                )
            )

            # =====================================================
            # GPU
            # =====================================================

            peak_gpu_util = (
                stats["peak_gpu"]
            )

            avg_gpu_util = (
                stats["avg_gpu"]
            )

            peak_gpu_memory = (
                stats["peak_vram"]
            )

            avg_gpu_memory = (
                stats["avg_vram"]
            )

            gpu_memory_total = (
                stats["vram_total"]
            )

            if gpu_memory_total > 0:

                peak_gpu_status = (

                    f"GPU "
                    f"{peak_gpu_util:.0f}% | "
                    f"VRAM "
                    f"{peak_gpu_memory:.0f}/"
                    f"{gpu_memory_total:.0f} MB"
                )

            else:

                peak_gpu_status = (

                    f"GPU "
                    f"{peak_gpu_util:.0f}% | "
                    f"VRAM N/A"
                )

            # =====================================================
            # CPU
            # =====================================================

            peak_cpu = (
                stats["peak_cpu"]
            )

            avg_cpu = (
                stats["avg_cpu"]
            )

            peak_cpu_status = (
                f"{peak_cpu:.1f}%"
            )

            # =====================================================
            # RAM
            # =====================================================

            peak_ram = (
                stats["peak_ram"]
            )

            avg_ram = (
                stats["avg_ram"]
            )

            peak_ram_percent = (
                stats["peak_ram_percent"]
            )

            avg_ram_percent = (
                stats["avg_ram_percent"]
            )

            peak_ram_status = (

                f"{peak_ram:.2f} GB "
                f"({peak_ram_percent:.1f}%)"
            )

            # =====================================================
            # STATUS
            # =====================================================

            if (
                result is not None
                and result.returncode == 0
            ):

                status = "SUCCESS"

            else:

                status = "ERROR"

            # =====================================================
            # DISPLAY RESULTS
            # =====================================================

            self.output_area.append(
                f"STATUS: {status}"
            )

            self.output_area.append(
                f"Output Resolution: "
                f"{output_resolution}"
            )

            self.output_area.append(
                f"Runtime: "
                f"{runtime_string}"
            )

            self.output_area.append(
                f"Processing FPS: "
                f"{processing_fps:.2f}"
            )

            self.output_area.append(
                f"Peak GPU: "
                f"{peak_gpu_status}"
            )

            self.output_area.append(
                f"Average GPU: "
                f"{avg_gpu_util:.2f}%"
            )

            self.output_area.append(
                f"Peak VRAM: "
                f"{peak_gpu_memory:.0f} MB"
            )

            self.output_area.append(
                f"Average VRAM: "
                f"{avg_gpu_memory:.0f} MB"
            )

            self.output_area.append(
                f"Peak CPU: "
                f"{peak_cpu_status}"
            )

            self.output_area.append(
                f"Average CPU: "
                f"{avg_cpu:.2f}%"
            )

            self.output_area.append(
                f"Peak RAM: "
                f"{peak_ram_status}"
            )

            self.output_area.append(
                f"Average RAM: "
                f"{avg_ram:.2f} GB "
                f"({avg_ram_percent:.1f}%)"
            )

            self.output_area.append(
                f"GPU Samples: "
                f"{stats['sample_count']}"
            )

            # =====================================================
            # CSV ROW
            # =====================================================

            csv_rows.append({

                "INPUT":
                    file_path,

                "OUTPUT":
                    output_dir,

                "ACTUAL RESOLUTION":
                    actual_resolution,

                "OUTPUT RESOLUTION":
                    output_resolution,

                "FPS":
                    f"{fps:.2f}",

                "FRAMES":
                    frame_count,

                "RUNTIME":
                    runtime_string,

                "PROCESSING FPS":
                    f"{processing_fps:.2f}",

                "PEAK GPU STATUS":
                    peak_gpu_status,

                "AVG GPU %":
                    f"{avg_gpu_util:.2f}",

                "PEAK VRAM MB":
                    f"{peak_gpu_memory:.0f}",

                "AVG VRAM MB":
                    f"{avg_gpu_memory:.0f}",

                "PEAK CPU STATUS":
                    peak_cpu_status,

                "AVG CPU %":
                    f"{avg_cpu:.2f}",

                "PEAK RAM STATUS":
                    peak_ram_status,

                "AVG RAM GB":
                    f"{avg_ram:.2f}",

                "GPU SAMPLES":
                    stats["sample_count"]
            })

            # =====================================================
            # TOTALS
            # =====================================================

            total_frames += (
                frame_count
            )

            total_runtime += (
                runtime_seconds
            )

            self.output_area.append(
                "=" * 70
            )

            self.output_area.append("")

        # =========================================================
        # SAVE CSV
        # =========================================================

        self.save_csv(
            csv_rows,
            output_dir,
            total_frames,
            total_runtime
        )

        self.button.setEnabled(
            True
        )

    # =============================================================
    # FORMAT RUNTIME
    # =============================================================

    def format_runtime(
        self,
        seconds
    ):

        seconds = int(
            seconds
        )

        hours = (
            seconds // 3600
        )

        minutes = (
            (seconds % 3600) // 60
        )

        secs = (
            seconds % 60
        )

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{secs:02d}"
        )

    # =============================================================
    # SAVE CSV
    # =============================================================

    def save_csv(
        self,
        rows,
        output_dir,
        total_frames,
        total_runtime
    ):

        csv_path = os.path.join(
            output_dir,
            "DepthCrafter_report.csv"
        )

        # ---------------------------------------------------------
        # Columns
        # ---------------------------------------------------------

        fieldnames = [

            "INPUT",

            "OUTPUT",

            "ACTUAL RESOLUTION",

            "OUTPUT RESOLUTION",

            "FPS",

            "FRAMES",

            "RUNTIME",

            "PROCESSING FPS",

            "PEAK GPU STATUS",

            "AVG GPU %",

            "PEAK VRAM MB",

            "AVG VRAM MB",

            "PEAK CPU STATUS",

            "AVG CPU %",

            "PEAK RAM STATUS",

            "AVG RAM GB",

            "GPU SAMPLES"
        ]

        # ---------------------------------------------------------
        # Total runtime
        # ---------------------------------------------------------

        total_runtime_string = (
            self.format_runtime(
                total_runtime
            )
        )

        try:

            with open(
                csv_path,
                "w",
                newline="",
                encoding="utf-8-sig"
            ) as csv_file:

                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=fieldnames
                )

                # -------------------------------------------------
                # Header
                # -------------------------------------------------

                writer.writeheader()

                # -------------------------------------------------
                # Video rows
                # -------------------------------------------------

                for row in rows:

                    writer.writerow(
                        row
                    )

                # -------------------------------------------------
                # Empty row
                # -------------------------------------------------

                writer.writerow({

                    column: ""

                    for column in fieldnames
                })

                # -------------------------------------------------
                # TOTAL FRAMES
                # -------------------------------------------------

                writer.writerow({

                    "INPUT":
                        "TOTAL FRAMES",

                    "FRAMES":
                        total_frames
                })

                # -------------------------------------------------
                # TOTAL RUNTIME
                # -------------------------------------------------

                writer.writerow({

                    "INPUT":
                        "TOTAL RUNTIME",

                    "RUNTIME":
                        total_runtime_string
                })

            # =====================================================
            # UI
            # =====================================================

            self.output_area.append(
                "\n" + "=" * 70
            )

            self.output_area.append(
                "CSV REPORT SAVED:"
            )

            self.output_area.append(
                csv_path
            )

            self.output_area.append(
                f"Total Frames: "
                f"{total_frames}"
            )

            self.output_area.append(
                f"Total Runtime: "
                f"{total_runtime_string}"
            )

            self.output_area.append(
                "=" * 70
            )

            QMessageBox.information(

                self,

                "Completed",

                "Processing completed.\n\n"
                f"CSV report:\n{csv_path}"
            )

        except Exception as e:

            QMessageBox.critical(

                self,

                "CSV Error",

                f"Could not save CSV:\n{str(e)}"
            )

    # =============================================================
    # GET OUTPUT RESOLUTION
    # =============================================================

    def get_output_resolution(
        self,
        input_file,
        output_dir,
        process_start_time
    ):
        """
        Find generated output files and determine resolution.

        Only files created/modified after the current process
        started are considered when possible.
        """

        input_name = Path(
            input_file
        ).stem

        image_extensions = (

            ".png",
            ".jpg",
            ".jpeg",
            ".exr",
            ".tif",
            ".tiff"
        )

        video_extensions = (

            ".mp4",
            ".mov",
            ".avi",
            ".mkv"
        )

        candidates = []

        # =========================================================
        # SEARCH OUTPUT DIRECTORY
        # =========================================================

        for root, dirs, files in os.walk(
            output_dir
        ):

            for file in files:

                full_path = os.path.join(
                    root,
                    file
                )

                ext = Path(
                    file
                ).suffix.lower()

                if ext not in (
                    image_extensions +
                    video_extensions
                ):

                    continue

                # -------------------------------------------------
                # Prefer newly generated files.
                # -------------------------------------------------

                try:

                    modified_time = os.path.getmtime(
                        full_path
                    )

                except Exception:

                    modified_time = 0

                # -------------------------------------------------
                # Prefer filename matching input.
                # -------------------------------------------------

                priority = 0

                if input_name.lower() in file.lower():

                    priority += 10

                if modified_time >= process_start_time:

                    priority += 20

                candidates.append(
                    (
                        priority,
                        modified_time,
                        full_path
                    )
                )

        if not candidates:

            return "N/A"

        # =========================================================
        # SORT CANDIDATES
        #
        # Highest priority first.
        # =========================================================

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1]
            ),
            reverse=True
        )

        # =========================================================
        # CHECK FILES
        # =========================================================

        for priority, modified_time, output_file in candidates:

            try:

                ext = Path(
                    output_file
                ).suffix.lower()

                # -------------------------------------------------
                # VIDEO
                # -------------------------------------------------

                if ext in video_extensions:

                    cap = cv2.VideoCapture(
                        output_file
                    )

                    if cap.isOpened():

                        width = int(
                            cap.get(
                                cv2.CAP_PROP_FRAME_WIDTH
                            )
                        )

                        height = int(
                            cap.get(
                                cv2.CAP_PROP_FRAME_HEIGHT
                            )
                        )

                        cap.release()

                        if (
                            width > 0
                            and height > 0
                        ):

                            return (
                                f"{width}x"
                                f"{height}"
                            )

                # -------------------------------------------------
                # IMAGE
                # -------------------------------------------------

                else:

                    image = cv2.imread(
                        output_file,
                        cv2.IMREAD_UNCHANGED
                    )

                    if image is not None:

                        height, width = (
                            image.shape[:2]
                        )

                        if (
                            width > 0
                            and height > 0
                        ):

                            return (
                                f"{width}x"
                                f"{height}"
                            )

            except Exception:

                continue

        return "N/A"

    # =============================================================
    # GPU STATUS
    # =============================================================

    def get_gpu_status(self):

        """
        Gets current NVIDIA GPU status.

        This is only a snapshot.

        The CSV PEAK GPU values come from SystemMonitor,
        which continuously samples during inference.
        """

        try:

            result = subprocess.run(
                [
                    "nvidia-smi",

                    "--query-gpu="
                    "utilization.gpu,"
                    "memory.used,"
                    "memory.total",

                    "--format="
                    "csv,noheader,nounits"
                ],

                capture_output=True,

                text=True,

                shell=False,

                timeout=0.5
            )

            if result.returncode != 0:

                return "N/A"

            output = (
                result.stdout
                .strip()
            )

            if not output:

                return "N/A"

            gpu_values = (
                output.split(",")
            )

            if len(gpu_values) >= 3:

                gpu_util = (
                    gpu_values[0].strip()
                )

                memory_used = (
                    gpu_values[1].strip()
                )

                memory_total = (
                    gpu_values[2].strip()
                )

                return (

                    f"GPU {gpu_util}% | "
                    f"VRAM "
                    f"{memory_used}/"
                    f"{memory_total} MB"
                )

            return output

        except Exception:

            return "N/A"

    # =============================================================
    # BROWSE INPUT
    # =============================================================

    def browse_file(self):

        file_path = (
            QFileDialog.getExistingDirectory(
                self,
                "Select Folder"
            )
        )

        if file_path:

            self.path_field.setText(
                file_path
            )

    # =============================================================
    # BROWSE OUTPUT
    # =============================================================

    def browse_output_path(self):

        file_path = (
            QFileDialog.getExistingDirectory(
                self,
                "Select Folder"
            )
        )

        if file_path:

            self.output_field.setText(
                file_path
            )


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = DepthCrafter()

    window.show()

    sys.exit(
        app.exec_()
    )