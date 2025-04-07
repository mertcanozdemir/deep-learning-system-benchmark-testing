# 🧠 Deep Learning System Benchmark

[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()
[![Build Status](https://github.com/mertcanozdemir/deep-learning-system-benchmark/actions/workflows/python-app.yml/badge.svg)](https://github.com/mertcanozdemir/deep-learning-system-benchmark/actions)

## 📌 Overview

**Deep Learning System Benchmark** is a Python-based benchmarking suite designed to assess your system's performance for deep learning workflows. It supports:

- ✅ CPU & GPU performance (PyTorch & TensorFlow)
- ✅ Matrix multiplication, convolution, SVD ops
- ✅ Model training and inference speed
- ✅ Memory allocation, copy bandwidth
- ✅ Disk read/write throughput
- ✅ CUDA-specific metrics (Tensor Cores, transfers, precision)

It generates a full report + performance plots — perfect for comparing environments, checking bottlenecks, or validating hardware upgrades.

---

## 🚀 Installation

```bash
git clone https://github.com/mertcanozdemir/deep-learning-system-benchmark.git
cd deep-learning-system-benchmark
pip install -r requirements.txt
```

> Optional: Install GPU libraries like `torch`, `tensorflow`, `GPUtil`, `py-cpuinfo` for full capabilities.

---

## 🧪 Running the Benchmark

```bash
python benchmark.py
```

To skip plot generation:

```bash
python benchmark.py --no-plots
```

Specify a custom output directory:

```bash
python benchmark.py --output-dir my_results/
```

---

## 📊 Output

- ✅ JSON log of all results (`benchmark_results/`)
- ✅ Text report summary
- ✅ Plot visualizations: matrix ops, I/O, CUDA precision

Example:

```
benchmark_results/
├── benchmark_results_20250408_143021.json
├── benchmark_report_20250408_143021.txt
└── plots/
    ├── matrix_multiplication.png
    ├── inference_performance.png
    ├── cuda_precision.png
    └── disk_performance.png
```

---

## 📎 Requirements

- Python ≥ 3.7  
- NumPy, pandas, matplotlib  
- Optional: PyTorch, TensorFlow, GPUtil, psutil, cpuinfo  

Install extras with:

```bash
pip install torch tensorflow gputil py-cpuinfo psutil
```

---

## 🛠 Features

- 🔬 Compare CPU vs GPU performance
- 🔁 Measure TensorCore acceleration & mixed precision
- 📉 Profile I/O bottlenecks and memory limits
- 📥 Export results for tracking across machines

---

## 📄 License

MIT License © [Your Name](https://github.com/mertcanozdemir)

---

## 🤝 Contributing

Pull requests are welcome! If you find a bug or have a feature idea, please [open an issue](https://github.com/mertcanozdemir/deep-learning-system-benchmark/issues).
