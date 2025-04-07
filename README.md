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
Example text output:

System: Windows 10
CPU: AMD64 (8 physical / 16 logical cores)
Memory: 15.37 GB
GPU: NVIDIA GeForce GTX 1650 Ti (4.0 GB CUDA)

Overall Rating: 10/10
Category: Excellent for deep learning

🧠 Deep Learning Performance (PyTorch):
- ResNet50 inference: 214.4 images/sec
- CNN inference: 236,845 images/sec
- CNN training: 400.3 iterations/sec
- Matrix multiplication (1024x1024): 0.0010 sec

🧠 Deep Learning Performance (TensorFlow):
- ResNet50 inference: 23.0 images/sec
- CNN inference: 1,961.7 images/sec
- CNN training: 0.025 sec/epoch
- Matrix multiplication (1024x1024): 0.0059 sec

🔢 NumPy:
- Matrix multiplication (1024x1024): 0.0050 sec
- SVD (512x512): 0.307 sec

💾 Disk I/O:
- Read speed: 2433 MB/s
- Write speed: 92.6 MB/s

🧬 Memory:
- Estimated bandwidth: 2511 MB/s
- Copy speed (1GB): 2631 MB/s

⚠️ Weakness:
- Slow CPU-GPU transfers (4.9 GB/s)

💡 Recommendation:
- Consider upgrading RAM to at least 16GB for better performance with larger models.

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
