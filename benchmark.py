import os
import time
import json
import platform
import argparse
import subprocess
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
# Set matplotlib backend to non-GUI 'Agg' to avoid Qt dependency issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import psutil

# Try importing deep learning libraries with graceful fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Some benchmarks will be skipped.")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Some benchmarks will be skipped.")

try:
    import cpuinfo
    CPU_INFO_AVAILABLE = True
except ImportError:
    CPU_INFO_AVAILABLE = False
    print("py-cpuinfo not available. CPU details will be limited.")

try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False
    print("GPUtil not available. NVIDIA GPU details will be limited.")



class DeepLearningBenchmark:
    def __init__(self, output_dir="benchmark_results", no_plots=False):
        self.results = {
            "system_info": {},
            "pytorch": {},
            "tensorflow": {},
            "numpy": {},
            "memory": {},
            "disk": {},
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.no_plots = no_plots
        
    def run_all_benchmarks(self):
        """Run all benchmarks and save results"""
        print("🔍 Gathering system information...")
        self.gather_system_info()
        
        print("\n📊 Running NumPy benchmarks...")
        self.benchmark_numpy()
        
        if TORCH_AVAILABLE:
            print("\n🔥 Running PyTorch benchmarks...")
            self.benchmark_pytorch()
        
        if TF_AVAILABLE:
            print("\n🧠 Running TensorFlow benchmarks...")
            self.benchmark_tensorflow()
        
        print("\n💾 Checking storage performance...")
        self.benchmark_disk_io()
        
        print("\n💻 Checking memory performance...")
        self.benchmark_memory()
        
        print("\n📝 Saving results...")
        self.save_results()
        
        print("\n📈 Generating performance report...")
        # Pass the no_plots flag from command line arguments
        self.generate_report(no_plots=hasattr(self, 'no_plots') and self.no_plots)
        
        print("\n✅ Benchmarks completed!")
        self.print_summary()
        
    def gather_system_info(self):
        """Gather detailed system information"""
        system_info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
        }
        
        # Get detailed CPU info if available
        if CPU_INFO_AVAILABLE:
            cpu_info = cpuinfo.get_cpu_info()
            system_info.update({
                "cpu_brand": cpu_info.get("brand_raw", "Unknown"),
                "cpu_hz": cpu_info.get("hz_actual_friendly", "Unknown"),
                "cpu_arch": cpu_info.get("arch", "Unknown"),
                "cpu_bits": cpu_info.get("bits", "Unknown"),
                "cpu_flags": cpu_info.get("flags", [])
            })
        
        # PyTorch GPU info
        if TORCH_AVAILABLE:
            system_info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                system_info["cuda_version"] = torch.version.cuda
                system_info["cuda_device_count"] = torch.cuda.device_count()
                system_info["cuda_current_device"] = torch.cuda.current_device()
                system_info["cuda_device_name"] = torch.cuda.get_device_name(0)
                
                # Get memory info for the GPU
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    system_info[f"cuda_device_{i}_name"] = props.name
                    system_info[f"cuda_device_{i}_total_memory"] = round(props.total_memory / (1024**3), 2)  # GB
        
        # TensorFlow GPU info
        if TF_AVAILABLE:
            system_info["tf_version"] = tf.__version__
            system_info["tf_gpu_available"] = len(tf.config.list_physical_devices('GPU')) > 0
            system_info["tf_physical_devices"] = {
                device_type: len(tf.config.list_physical_devices(device_type))
                for device_type in ['GPU', 'CPU']
            }
            
            # Get more detailed TF GPU info if available
            if system_info["tf_gpu_available"]:
                gpus = tf.config.list_physical_devices('GPU')
                for i, gpu in enumerate(gpus):
                    system_info[f"tf_gpu_{i}"] = str(gpu)
        
        # Additional GPU info using GPUtil for NVIDIA GPUs
        if GPU_UTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    system_info[f"gpu_{i}_name"] = gpu.name
                    system_info[f"gpu_{i}_driver"] = gpu.driver
                    system_info[f"gpu_{i}_memory_total"] = gpu.memoryTotal
                    system_info[f"gpu_{i}_memory_used"] = gpu.memoryUsed
                    system_info[f"gpu_{i}_temperature"] = gpu.temperature
            except Exception as e:
                system_info["gputil_error"] = str(e)
        
        self.results["system_info"] = system_info
        
        # Print some basic system info
        print(f"System: {system_info['platform']}")
        print(f"CPU: {system_info.get('cpu_brand', system_info['processor'])}")
        print(f"Cores: {system_info['physical_cpu_count']} physical, {system_info['logical_cpu_count']} logical")
        print(f"Memory: {system_info['total_memory_gb']} GB")
        
        if TORCH_AVAILABLE and system_info["cuda_available"]:
            print(f"GPU: {system_info['cuda_device_name']}")
            print(f"CUDA Version: {system_info['cuda_version']}")
        elif TF_AVAILABLE and system_info["tf_gpu_available"]:
            print(f"TensorFlow GPU: {system_info['tf_physical_devices']}")
        else:
            print("No GPU detected for deep learning")
    
    def benchmark_numpy(self):
        """Benchmark NumPy for basic linear algebra operations"""
        print("Testing NumPy matrix operations...")
        results = {}
        
        # Test matrix multiplication with increasing sizes
        matrix_sizes = [128, 512, 1024, 2048, 4096]
        for size in matrix_sizes:
            print(f"  Matrix multiplication {size}x{size}...", end="", flush=True)
            
            # Generate random matrices
            a = np.random.rand(size, size).astype(np.float32)
            b = np.random.rand(size, size).astype(np.float32)
            
            # Warm-up
            np.matmul(a, b)
            
            # Benchmark
            start_time = time.time()
            np.matmul(a, b)
            elapsed = time.time() - start_time
            
            results[f"matmul_{size}"] = elapsed
            print(f" {elapsed:.3f} seconds")
        
        # Test SVD decomposition (common in many ML algorithms)
        for size in [128, 512, 1024]:
            print(f"  SVD decomposition {size}x{size}...", end="", flush=True)
            
            # Generate random matrix
            a = np.random.rand(size, size).astype(np.float32)
            
            # Benchmark
            start_time = time.time()
            np.linalg.svd(a)
            elapsed = time.time() - start_time
            
            results[f"svd_{size}"] = elapsed
            print(f" {elapsed:.3f} seconds")
        
        self.results["numpy"] = results

    def benchmark_pytorch(self):
        """Benchmark PyTorch operations and models"""
        if not TORCH_AVAILABLE:
            return
        
        results = {
            "operations": {},
            "training": {},
            "inference": {},
            "cuda": {}  # New section for CUDA-specific tests
        }
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running PyTorch benchmarks on: {device}")
        
        # Run CUDA-specific tests if available
        if device.type == "cuda":
            print("Running CUDA-specific tests...")
            cuda_results = self._benchmark_cuda_operations()
            results["cuda"] = cuda_results
        
        # Test tensor operations
        print("Testing PyTorch tensor operations...")
        
        # Matrix multiplication
        matrix_sizes = [512, 1024, 2048, 4096]
        for size in matrix_sizes:
            print(f"  Matrix multiplication {size}x{size}...", end="", flush=True)
            
            # Generate random matrices
            a = torch.rand(size, size, device=device)
            b = torch.rand(size, size, device=device)
            
            # Warm-up
            torch.matmul(a, b)
            torch.cuda.synchronize() if device.type == "cuda" else None
            
            # Benchmark
            start_time = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize() if device.type == "cuda" else None
            elapsed = time.time() - start_time
            
            results["operations"][f"matmul_{size}"] = elapsed
            print(f" {elapsed:.3f} seconds")
        
        # Convolution operation (important for CNNs)
        batch_sizes = [16, 32, 64]
        for batch_size in batch_sizes:
            print(f"  Conv2d batch_size={batch_size}...", end="", flush=True)
            
            # Create a sample input and convolution layer
            input_tensor = torch.rand(batch_size, 3, 224, 224, device=device)
            conv_layer = nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)
            
            # Warm-up
            conv_layer(input_tensor)
            torch.cuda.synchronize() if device.type == "cuda" else None
            
            # Benchmark
            start_time = time.time()
            output = conv_layer(input_tensor)
            torch.cuda.synchronize() if device.type == "cuda" else None
            elapsed = time.time() - start_time
            
            results["operations"][f"conv2d_b{batch_size}"] = elapsed
            print(f" {elapsed:.3f} seconds")
        
        # Test model training (simple CNN for MNIST-like data)
        print("Testing model training...")
        
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 16, 3, 1)
                self.conv2 = nn.Conv2d(16, 32, 3, 1)
                self.fc1 = nn.Linear(5*5*32, 128)
                self.fc2 = nn.Linear(128, 10)
                self.relu = nn.ReLU()
                self.max_pool = nn.MaxPool2d(2)
                
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.max_pool(x)
                x = self.relu(self.conv2(x))
                x = self.max_pool(x)
                x = x.view(-1, 5*5*32)
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        # Create a simple dataset
        batch_size = 64
        train_data = torch.rand(batch_size, 1, 28, 28, device=device)
        train_labels = torch.randint(0, 10, (batch_size,), device=device)
        
        # Create model and optimizer
        model = SimpleCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Warm-up
        for _ in range(3):
            optimizer.zero_grad()
            output = model(train_data)
            loss = criterion(output, train_labels)
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        
        # Benchmark training for 10 iterations
        start_time = time.time()
        for _ in range(10):
            optimizer.zero_grad()
            output = model(train_data)
            loss = criterion(output, train_labels)
            loss.backward()
            optimizer.step()
            
        torch.cuda.synchronize() if device.type == "cuda" else None
        train_elapsed = time.time() - start_time
        
        results["training"]["cnn_10epochs"] = train_elapsed
        results["training"]["cnn_iterations_per_sec"] = 10 / train_elapsed
        print(f"  Training 10 iterations: {train_elapsed:.3f} seconds ({10/train_elapsed:.2f} it/s)")
        
        # Test inference with larger batch size
        print("Testing inference...")
        
        infer_batch_size = 128
        infer_data = torch.rand(infer_batch_size, 1, 28, 28, device=device)
        
        # Warm-up
        with torch.no_grad():
            model(infer_data)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        
        # Benchmark inference
        n_runs = 50
        start_time = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                model(infer_data)
                
        torch.cuda.synchronize() if device.type == "cuda" else None
        infer_elapsed = time.time() - start_time
        
        results["inference"]["cnn_inference"] = infer_elapsed
        results["inference"]["images_per_sec"] = (n_runs * infer_batch_size) / infer_elapsed
        print(f"  Inference: {infer_elapsed:.3f} seconds " +
              f"({(n_runs * infer_batch_size) / infer_elapsed:.2f} images/s)")
        
        # Test with a larger model (ResNet-like)
        if device.type == "cuda" or self.results["system_info"]["total_memory_gb"] > 8:
            print("Testing larger model (ResNet)...")
            
            # Try importing a pretrained model
            try:
                from torchvision.models import resnet50
                resnet = resnet50(pretrained=False).to(device)
                resnet.eval()
                
                # Benchmark with standard ImageNet size
                resnet_batch = 16
                resnet_data = torch.rand(resnet_batch, 3, 224, 224, device=device)
                
                # Warm-up
                with torch.no_grad():
                    resnet(resnet_data)
                    
                torch.cuda.synchronize() if device.type == "cuda" else None
                
                # Benchmark
                resnet_runs = 10
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(resnet_runs):
                        resnet(resnet_data)
                        
                torch.cuda.synchronize() if device.type == "cuda" else None
                resnet_elapsed = time.time() - start_time
                
                results["inference"]["resnet50_inference"] = resnet_elapsed
                results["inference"]["resnet50_images_per_sec"] = (resnet_runs * resnet_batch) / resnet_elapsed
                print(f"  ResNet50 inference: {resnet_elapsed:.3f} seconds " +
                      f"({(resnet_runs * resnet_batch) / resnet_elapsed:.2f} images/s)")
                
            except Exception as e:
                print(f"  Could not test ResNet: {e}")
                results["inference"]["resnet50_error"] = str(e)
        
        self.results["pytorch"] = results

    def benchmark_tensorflow(self):
        """Benchmark TensorFlow operations and models"""
        if not TF_AVAILABLE:
            return
    
        results = {
        "operations": {},
        "training": {},
        "inference": {}
    }
    
        print("Testing TensorFlow tensor operations...")
    
    # Check if GPU is being used
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"TensorFlow using GPU: {physical_devices[0]}")
        else:
            print("TensorFlow using CPU")
    
    # Matrix multiplication
        matrix_sizes = [512, 1024, 2048, 4096]
        for size in matrix_sizes:
            print(f"  Matrix multiplication {size}x{size}...", end="", flush=True)
        
        # Generate random matrices
            a = tf.random.normal([size, size])
            b = tf.random.normal([size, size])
        
        # Warm-up
            tf.matmul(a, b)
        
        # Benchmark
            start_time = time.time()
            c = tf.matmul(a, b)
            elapsed = time.time() - start_time
        
            results["operations"][f"matmul_{size}"] = elapsed
            print(f" {elapsed:.3f} seconds")
    
    # Convolution operation
        batch_sizes = [16, 32, 64]
        for batch_size in batch_sizes:
            print(f"  Conv2D batch_size={batch_size}...", end="", flush=True)
        
        # Create a sample input and convolution layer
            input_tensor = tf.random.normal([batch_size, 224, 224, 3])
            conv_layer = tf.keras.layers.Conv2D(64, 3, padding='same')
        
        # Warm-up
            _ = conv_layer(input_tensor)
        
        # Benchmark
            start_time = time.time()
            output = conv_layer(input_tensor)
            elapsed = time.time() - start_time
        
            results["operations"][f"conv2d_b{batch_size}"] = elapsed
            print(f" {elapsed:.3f} seconds")
    
    # Test model training
        print("Testing model training...")
    
    # Create a simple CNN model
        model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    
        model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Create a simple dataset
        batch_size = 64
        train_data = tf.random.normal([batch_size, 28, 28, 1])
        train_labels = tf.random.uniform([batch_size], minval=0, maxval=10, dtype=tf.int32)
    
    # Warm-up
        model.fit(train_data, train_labels, epochs=1, verbose=0)
    
    # Benchmark training
        start_time = time.time()
        model.fit(train_data, train_labels, epochs=5, verbose=0)
        train_elapsed = time.time() - start_time
    
        results["training"]["cnn_5epochs"] = train_elapsed
        results["training"]["seconds_per_epoch"] = train_elapsed / 5
        print(f"  Training 5 epochs: {train_elapsed:.3f} seconds ({train_elapsed/5:.3f} s/epoch)")
    
    # Test inference
        print("Testing inference...")
    
        infer_batch_size = 128
        infer_data = tf.random.normal([infer_batch_size, 28, 28, 1])
    
    # Warm-up
        model.predict(infer_data, verbose=0)
    
    # Benchmark inference
        n_runs = 20
        start_time = time.time()
        for _ in range(n_runs):
            model.predict(infer_data, verbose=0)
        
        infer_elapsed = time.time() - start_time
    
        results["inference"]["cnn_inference"] = infer_elapsed
        results["inference"]["images_per_sec"] = (n_runs * infer_batch_size) / infer_elapsed
        print(f"  Inference: {infer_elapsed:.3f} seconds " +
            f"({(n_runs * infer_batch_size) / infer_elapsed:.2f} images/s)")
    
    # Test with a larger model if available
        try:
            if physical_devices or self.results["system_info"]["total_memory_gb"] > 8:
                print("Testing larger model (ResNet50)...")
            
            # Load pretrained model
                resnet = tf.keras.applications.ResNet50(weights=None, include_top=True)
            
            # Benchmark with standard ImageNet size
                resnet_batch = 16
                resnet_data = tf.random.normal([resnet_batch, 224, 224, 3])
            
            # Warm-up
                resnet.predict(resnet_data, verbose=0)
            
            # Benchmark
                resnet_runs = 10
                start_time = time.time()
                for _ in range(resnet_runs):
                    resnet.predict(resnet_data, verbose=0)
                
                resnet_elapsed = time.time() - start_time
            
                results["inference"]["resnet50_inference"] = resnet_elapsed
                results["inference"]["resnet50_images_per_sec"] = (resnet_runs * resnet_batch) / resnet_elapsed
                print(f"  ResNet50 inference: {resnet_elapsed:.3f} seconds " +
                    f"({(resnet_runs * resnet_batch) / resnet_elapsed:.2f} images/s)")
        except Exception as e:
            print(f"  Could not test ResNet: {e}")
            results["inference"]["resnet50_error"] = str(e)
    
        self.results["tensorflow"] = results

    def _benchmark_cuda_operations(self):
        """Run CUDA-specific benchmarks to test GPU capabilities"""
        cuda_results = {}
        
        try:
            # Check CUDA and GPU properties
            cuda_results["cuda_version"] = torch.version.cuda
            cuda_results["cudnn_version"] = torch.backends.cudnn.version()
            cuda_results["cudnn_enabled"] = torch.backends.cudnn.enabled
            cuda_results["gpu_count"] = torch.cuda.device_count()
            
            # Test CUDA memory operations
            print("  Testing CUDA memory operations...")
            mem_results = self._test_cuda_memory()
            cuda_results["memory"] = mem_results
            
            # Test CUDA transfer speeds
            print("  Testing CPU-GPU transfer speeds...")
            transfer_results = self._test_cuda_transfers()
            cuda_results["transfer"] = transfer_results
            
            # Test CUDA kernel launch overhead
            print("  Testing CUDA kernel launch overhead...")
            kernel_results = self._test_cuda_kernel_overhead()
            cuda_results["kernel_overhead"] = kernel_results
            
            # Test CUDA numerical precision
            print("  Testing CUDA numerical precision...")
            precision_results = self._test_cuda_precision()
            cuda_results["precision"] = precision_results
            
            # Test multi-GPU if available
            if torch.cuda.device_count() > 1:
                print(f"  Testing multi-GPU operations across {torch.cuda.device_count()} GPUs...")
                multi_gpu_results = self._test_multi_gpu()
                cuda_results["multi_gpu"] = multi_gpu_results
            
        except Exception as e:
            print(f"  Error during CUDA tests: {e}")
            cuda_results["error"] = str(e)
        
        return cuda_results
        
    def _test_cuda_memory(self):
        """Test CUDA memory allocation, transfer, and bandwidth"""
        results = {}
        
        # Get total and available memory
        results["total_memory"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
        results["max_memory_reserved"] = torch.cuda.max_memory_reserved() / (1024**3)  # in GB
        
        # Test allocation time for different tensor sizes
        sizes = [
            (1000, 1000),      # ~4 MB for float32
            (5000, 5000),      # ~100 MB for float32
            (10000, 10000),    # ~400 MB for float32
        ]
        
        alloc_times = {}
        for i, size in enumerate(sizes):
            # Skip larger tests if we're running low on memory
            if i > 0 and torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory > 0.7:
                print(f"    Skipping allocation test for size {size} due to memory constraints")
                continue
                
            # Measure allocation time
            torch.cuda.synchronize()
            start = time.time()
            x = torch.rand(size, device="cuda")
            torch.cuda.synchronize()
            alloc_time = time.time() - start
            
            # Store result and free memory
            size_mb = x.nelement() * x.element_size() / (1024**2)
            alloc_times[f"{size[0]}x{size[1]}"] = {
                "time_seconds": alloc_time,
                "size_mb": size_mb,
                "speed_gbps": (size_mb / 1024) / alloc_time
            }
            del x
            torch.cuda.empty_cache()
            
        results["allocation"] = alloc_times
        
        # Test memset speed (zeroing memory)
        memset_times = {}
        for size in [(10000, 10000)]:  # Just test one large size
            x = torch.rand(size, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            x.zero_()
            torch.cuda.synchronize()
            memset_time = time.time() - start
            
            size_mb = x.nelement() * x.element_size() / (1024**2)
            memset_times[f"{size[0]}x{size[1]}"] = {
                "time_seconds": memset_time,
                "size_mb": size_mb,
                "speed_gbps": (size_mb / 1024) / memset_time
            }
            del x
            torch.cuda.empty_cache()
            
        results["memset"] = memset_times
        
        return results
    
    def _test_cuda_transfers(self):
        """Test CPU to GPU and GPU to CPU transfer speeds"""
        results = {}
        
        sizes = [
            (1000, 1000),      # ~4 MB for float32
            (8000, 8000),      # ~256 MB for float32
        ]
        
        h2d_times = {}  # Host to Device (CPU to GPU)
        d2h_times = {}  # Device to Host (GPU to CPU)
        
        for size in sizes:
            # Create CPU tensor
            x_cpu = torch.rand(size)
            
            # Measure Host to Device transfer (CPU -> GPU)
            torch.cuda.synchronize()
            start = time.time()
            x_gpu = x_cpu.cuda()
            torch.cuda.synchronize()
            h2d_time = time.time() - start
            
            size_mb = x_cpu.nelement() * x_cpu.element_size() / (1024**2)
            h2d_times[f"{size[0]}x{size[1]}"] = {
                "time_seconds": h2d_time,
                "size_mb": size_mb,
                "speed_gbps": (size_mb / 1024) / h2d_time
            }
            
            # Measure Device to Host transfer (GPU -> CPU)
            torch.cuda.synchronize()
            start = time.time()
            x_back = x_gpu.cpu()
            torch.cuda.synchronize()
            d2h_time = time.time() - start
            
            d2h_times[f"{size[0]}x{size[1]}"] = {
                "time_seconds": d2h_time,
                "size_mb": size_mb,
                "speed_gbps": (size_mb / 1024) / d2h_time
            }
            
            # Clean up
            del x_cpu, x_gpu, x_back
            torch.cuda.empty_cache()
        
        results["cpu_to_gpu"] = h2d_times
        results["gpu_to_cpu"] = d2h_times
        
        # Test pinned memory transfer speed if available
        try:
            # Create pinned memory tensor
            x_pinned = torch.rand(8000, 8000).pin_memory()
            
            # Measure transfer with pinned memory
            torch.cuda.synchronize()
            start = time.time()
            x_gpu = x_pinned.cuda(non_blocking=True)
            torch.cuda.synchronize()
            pinned_time = time.time() - start
            
            size_mb = x_pinned.nelement() * x_pinned.element_size() / (1024**2)
            results["pinned_memory"] = {
                "time_seconds": pinned_time,
                "size_mb": size_mb,
                "speed_gbps": (size_mb / 1024) / pinned_time
            }
            
            # Clean up
            del x_pinned, x_gpu
            torch.cuda.empty_cache()
        except Exception as e:
            results["pinned_memory_error"] = str(e)
        
        return results
    
    def _test_cuda_kernel_overhead(self):
        """Test CUDA kernel launch overhead"""
        results = {}
        
        # Test with a simple operation (addition)
        x = torch.ones(10, device="cuda")
        y = torch.ones(10, device="cuda")
        
        # Warm up
        for _ in range(5):
            z = x + y
            
        torch.cuda.synchronize()
        
        # Measure overhead for many small kernel launches
        iterations = 1000
        start = time.time()
        for _ in range(iterations):
            z = x + y
            torch.cuda.synchronize()  # Force waiting for kernel completion
            
        total_time = time.time() - start
        
        results["small_kernel_avg_overhead_ms"] = (total_time / iterations) * 1000
        
        # Test kernel fusion capabilities
        # Create tensors for multiple operations
        a = torch.rand(5000, 5000, device="cuda")
        b = torch.rand(5000, 5000, device="cuda")
        c = torch.rand(5000, 5000, device="cuda")
        
        # Warm up
        d = a + b + c
        del d
        torch.cuda.synchronize()
        
        # Measure time for separate operations
        start = time.time()
        temp = a + b
        result = temp + c
        torch.cuda.synchronize()
        separate_time = time.time() - start
        
        # Measure time for fused operations
        torch.cuda.synchronize()
        start = time.time()
        result = a + b + c
        torch.cuda.synchronize()
        fused_time = time.time() - start
        
        results["separate_ops_time"] = separate_time
        results["fused_ops_time"] = fused_time
        results["fusion_speedup"] = separate_time / fused_time if fused_time > 0 else 0
        
        return results
    
    def _test_cuda_precision(self):
        """Test different numerical precision performance on CUDA"""
        results = {}
        
        # Test different data types
        dtypes = [
            (torch.float32, "float32"),
            (torch.float16, "float16"),
            (torch.int32, "int32"),
            (torch.int64, "int64")
        ]
        
        # Try to add bfloat16 if available
        try:
            dtypes.append((torch.bfloat16, "bfloat16"))
        except AttributeError:
            pass
        
        # Create tensors and test matmul performance
        size = 4096
        for dtype, name in dtypes:
            try:
                # Skip integer types for matmul
                if "int" in name:
                    continue
                    
                a = torch.rand(size, size, dtype=dtype, device="cuda")
                b = torch.rand(size, size, dtype=dtype, device="cuda")
                
                # Warm up
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                
                # Measure performance
                start = time.time()
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                elapsed = time.time() - start
                
                # Calculate throughput in TFLOPS
                # Each matrix multiply does 2*N^3 operations
                ops = 2 * (size ** 3)
                results[f"matmul_{name}_time"] = elapsed
                results[f"matmul_{name}_tflops"] = (ops / elapsed) / 1e12
                
                # Clean up
                del a, b, c
                torch.cuda.empty_cache()
            except Exception as e:
                results[f"error_{name}"] = str(e)
        
        # Test Tensor Core operations if available
        if torch.cuda.get_device_capability(0)[0] >= 7:  # Volta or later
            try:
                # Enable TensorCores
                torch.backends.cudnn.benchmark = True
                
                a = torch.rand(size, size, dtype=torch.float16, device="cuda")
                b = torch.rand(size, size, dtype=torch.float16, device="cuda")
                
                # Warm up with TensorCores
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                
                # Measure performance
                start = time.time()
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                tensorcore_time = time.time() - start
                
                # Calculate throughput
                ops = 2 * (size ** 3)
                results["matmul_tensorcore_time"] = tensorcore_time
                results["matmul_tensorcore_tflops"] = (ops / tensorcore_time) / 1e12
                
                # Clean up
                del a, b, c
                torch.cuda.empty_cache()
                
                # Reset benchmark setting
                torch.backends.cudnn.benchmark = False
            except Exception as e:
                results["error_tensorcore"] = str(e)
        
        return results
    
    def _test_multi_gpu(self):
        """Test multi-GPU operations if multiple GPUs are available"""
        results = {}
        gpu_count = torch.cuda.device_count()
        
        if gpu_count <= 1:
            return {"available": False}
            
        results["available"] = True
        results["count"] = gpu_count
        
        # Test data parallel performance
        try:
            from torch.nn.parallel import DataParallel
            
            class SimpleModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
                    self.bn = torch.nn.BatchNorm2d(64)
                    self.relu = torch.nn.ReLU()
                    self.fc = torch.nn.Linear(64 * 224 * 224, 1000)
                
                def forward(self, x):
                    x = self.relu(self.bn(self.conv(x)))
                    x = x.view(x.size(0), -1)
                    x = self.fc(x)
                    return x
            
            # Create model and wrap in DataParallel
            model = SimpleModel().cuda()
            dp_model = DataParallel(model)
            
            # Create input data
            batch_size = 16
            input_data = torch.rand(batch_size, 3, 224, 224, device="cuda")
            
            # Test single GPU
            torch.cuda.synchronize()
            start = time.time()
            output = model(input_data)
            torch.cuda.synchronize()
            single_gpu_time = time.time() - start
            
            # Test multi-GPU
            torch.cuda.synchronize()
            start = time.time()
            output = dp_model(input_data)
            torch.cuda.synchronize()
            multi_gpu_time = time.time() - start
            
            results["single_gpu_time"] = single_gpu_time
            results["multi_gpu_time"] = multi_gpu_time
            results["speedup"] = single_gpu_time / multi_gpu_time if multi_gpu_time > 0 else 0
            
            # Clean up
            del model, dp_model, input_data, output
            torch.cuda.empty_cache()
            
        except Exception as e:
            results["error"] = str(e)
        
        return results




    def print_summary(self):
            """Print a summary of benchmark results to console"""
            rating = self._calculate_performance_rating()
        
            print("\n" + "=" * 60)
            print(f"BENCHMARK SUMMARY")
            print("=" * 60)
            print(f"Overall Performance Rating: {rating['overall_rating']}/10")
            print(f"Category: {rating['category']}")
            print(f"\nStrengths: {', '.join(rating['strengths'])}")
            print(f"Weaknesses: {', '.join(rating['weaknesses'])}")
        
        # Add CUDA-specific summary if available
            if TORCH_AVAILABLE and torch.cuda.is_available() and "pytorch" in self.results and "cuda" in self.results["pytorch"]:
                cuda_results = self.results["pytorch"]["cuda"]
                if cuda_results and "precision" in cuda_results:
                    print("\nCUDA Performance:")
                
                # Show precision tests results if available
                    precision = cuda_results["precision"]
                    for key, value in precision.items():
                        if key.startswith("matmul_") and key.endswith("_tflops"):
                            format_name = key.replace("matmul_", "").replace("_tflops", "")
                            print(f"  {format_name.upper()} performance: {value:.2f} TFLOPS")
                
                # Show memory transfer speeds
                    if "transfer" in cuda_results and "cpu_to_gpu" in cuda_results["transfer"]:
                        for size, data in cuda_results["transfer"]["cpu_to_gpu"].items():
                            if "8000x8000" in size:  # Just show the largest size
                                print(f"  CPU → GPU transfer: {data.get('speed_gbps', 0):.2f} GB/s")
                                break
                
                # Show tensor cores if tested
                    if "matmul_tensorcore_tflops" in precision:
                        tc_speedup = precision.get("matmul_tensorcore_tflops", 0) / precision.get("matmul_float16_tflops", 1)
                        print(f"  Tensor Cores: {'Available' if tc_speedup > 1.2 else 'Limited benefit'}")
                        if tc_speedup > 1.2:
                            print(f"  Tensor Core speedup: {tc_speedup:.2f}x over FP16")
        
            print(f"\nRecommendation: {rating['recommendation']}")
            print("\nFull results and detailed report have been saved to:")
            print(f"  {self.report_file}")
            print("=" * 60)


    
    def _calculate_performance_rating(self):
        """Calculate an overall performance rating based on benchmark results"""
        rating = {
            "overall_rating": 0,
            "category": "",
            "strengths": [],
            "weaknesses": [],
            "recommendation": ""
        }
        
        # Default rating is average
        overall_score = 5.0
        
        # Check for GPU acceleration
        has_gpu = False
        gpu_score = 0
        
        if self.results["system_info"].get("cuda_available", False):
            has_gpu = True
            gpu_name = self.results["system_info"].get("cuda_device_name", "Unknown GPU")
            
            # Check CUDA performance if available
            cuda_performance = 0
            if "pytorch" in self.results and "cuda" in self.results["pytorch"]:
                cuda_results = self.results["pytorch"]["cuda"]
                if cuda_results and "precision" in cuda_results:
                    # Check FP32 performance
                    fp32_tflops = cuda_results["precision"].get("matmul_float32_tflops", 0)
                    if fp32_tflops > 30:
                        rating["strengths"].append(f"Exceptional GPU compute ({fp32_tflops:.1f} TFLOPS)")
                        cuda_performance = 3
                    elif fp32_tflops > 15:
                        rating["strengths"].append(f"Strong GPU compute ({fp32_tflops:.1f} TFLOPS)")
                        cuda_performance = 2
                    elif fp32_tflops > 5:
                        rating["strengths"].append(f"Good GPU compute ({fp32_tflops:.1f} TFLOPS)")
                        cuda_performance = 1
                    
                    # Check for tensor cores
                    if "matmul_tensorcore_tflops" in cuda_results["precision"]:
                        tc_tflops = cuda_results["precision"]["matmul_tensorcore_tflops"]
                        if tc_tflops > fp32_tflops * 2:
                            rating["strengths"].append(f"Tensor Cores ({tc_tflops:.1f} TFLOPS)")
                            cuda_performance += 1
                
                # Check memory transfer performance
                if "transfer" in cuda_results and "cpu_to_gpu" in cuda_results["transfer"]:
                    for size, data in cuda_results["transfer"]["cpu_to_gpu"].items():
                        if "8000x8000" in size:  # Just check the largest size
                            speed = data.get("speed_gbps", 0)
                            if speed > 12:
                                rating["strengths"].append(f"Fast CPU-GPU transfers ({speed:.1f} GB/s)")
                                cuda_performance += 0.5
                            elif speed < 5:
                                rating["weaknesses"].append(f"Slow CPU-GPU transfers ({speed:.1f} GB/s)")
                                cuda_performance -= 0.5
            
            gpu_score = 2 + cuda_performance
            
        elif self.results["system_info"].get("tf_gpu_available", False):
            has_gpu = True
            rating["strengths"].append("TensorFlow-compatible GPU")
            gpu_score = 1.5
        else:
            rating["weaknesses"].append("No GPU acceleration")
            gpu_score = -2
        
        # Check for memory
        memory_gb = self.results["system_info"].get("total_memory_gb", 0)
        if memory_gb >= 32:
            rating["strengths"].append(f"High RAM ({memory_gb} GB)")
            memory_score = 1.5
        elif memory_gb >= 16:
            rating["strengths"].append(f"Sufficient RAM ({memory_gb} GB)")
            memory_score = 1
        elif memory_gb >= 8:
            memory_score = 0
        else:
            rating["weaknesses"].append(f"Limited RAM ({memory_gb} GB)")
            memory_score = -1.5
        
        # Check CPU cores
        cpu_cores = self.results["system_info"].get("logical_cpu_count", 0)
        if cpu_cores >= 16:
            rating["strengths"].append(f"Many CPU cores ({cpu_cores})")
            cpu_score = 1
        elif cpu_cores >= 8:
            cpu_score = 0.5
        else:
            rating["weaknesses"].append(f"Limited CPU cores ({cpu_cores})")
            cpu_score = -0.5
        
        # Check inference performance if available
        inference_score = 0
        if "pytorch" in self.results and "inference" in self.results["pytorch"]:
            resnet_speed = self.results["pytorch"]["inference"].get("resnet50_images_per_sec", 0)
            if resnet_speed > 0:
                if resnet_speed > 100:
                    rating["strengths"].append(f"Fast ResNet inference ({resnet_speed:.1f} img/s)")
                    inference_score = 2
                elif resnet_speed > 50:
                    rating["strengths"].append(f"Good ResNet inference ({resnet_speed:.1f} img/s)")
                    inference_score = 1
                elif resnet_speed > 10:
                    inference_score = 0
                else:
                    rating["weaknesses"].append(f"Slow ResNet inference ({resnet_speed:.1f} img/s)")
                    inference_score = -1
        
        # Check disk I/O
        io_score = 0
        if "disk" in self.results:
            read_speed = self.results["disk"].get("read_speed_1000mb", 0)
            if read_speed > 500:
                rating["strengths"].append(f"Fast disk read ({read_speed:.1f} MB/s)")
                io_score = 1
            elif read_speed < 100:
                rating["weaknesses"].append(f"Slow disk read ({read_speed:.1f} MB/s)")
                io_score = -1
        
        # Calculate overall score
        overall_score = 5.0 + gpu_score + memory_score + cpu_score + inference_score + io_score
        
        # Ensure the score is between 1 and 10
        overall_score = max(1, min(10, overall_score))
        rating["overall_rating"] = round(overall_score, 1)
        
        # Determine category
        if overall_score >= 8.5:
            rating["category"] = "Excellent for deep learning"
        elif overall_score >= 7:
            rating["category"] = "Good for deep learning"
        elif overall_score >= 5:
            rating["category"] = "Adequate for deep learning"
        else:
            rating["category"] = "Limited for deep learning"
        
        # Generate recommendation
        if not has_gpu:
            rating["recommendation"] = "Consider adding a dedicated GPU to significantly improve deep learning performance."
        elif memory_gb < 16:
            rating["recommendation"] = "Consider upgrading RAM to at least 16GB for better performance with larger models."
        elif overall_score < 7:
            rating["recommendation"] = "Your system can run basic deep learning tasks, but may struggle with large models or datasets."
        else:
            rating["recommendation"] = "Your system is well-equipped for deep learning. Focus on optimizing your models and workflows."
        
        return rating    
    def _generate_plots(self):
        """Generate performance visualization plots"""
        try:
            # Create a plots directory
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            print("  Generating matrix multiplication plots...")
            # Plot matrix multiplication performance comparisons
            self._plot_matrix_multiplication(plots_dir)
            
            print("  Generating inference performance plots...")
            # Plot inference performance
            self._plot_inference_performance(plots_dir)
            
            print("  Generating I/O performance plots...")
            # Plot memory and disk performance
            self._plot_io_performance(plots_dir)
            
            # Plot CUDA-specific performance if available
            if torch.cuda.is_available():
                print("  Generating CUDA performance plots...")
                self._plot_cuda_performance(plots_dir)
            
            print(f"  Plots saved to: {plots_dir}")
        except Exception as e:
            print(f"Error generating plots: {e}")
            print("Try running with --no-plots if you continue to experience issues.")
    
    def _plot_matrix_multiplication(self, plots_dir):
        """Plot matrix multiplication performance comparison"""
        plt.figure(figsize=(10, 6))
        
        sizes = [512, 1024, 2048]
        np_times = []
        pt_times = []
        tf_times = []
        
        # Get NumPy times
        if "numpy" in self.results:
            for size in sizes:
                np_times.append(self.results["numpy"].get(f"matmul_{size}", 0))
        
        # Get PyTorch times
        if "pytorch" in self.results and "operations" in self.results["pytorch"]:
            for size in sizes:
                pt_times.append(self.results["pytorch"]["operations"].get(f"matmul_{size}", 0))
        
        # Get TensorFlow times
        if "tensorflow" in self.results and "operations" in self.results["tensorflow"]:
            for size in sizes:
                tf_times.append(self.results["tensorflow"]["operations"].get(f"matmul_{size}", 0))
        
        # Create the bar chart
        x = np.arange(len(sizes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if np_times:
            ax.bar(x - width, np_times, width, label='NumPy')
        if pt_times:
            ax.bar(x, pt_times, width, label='PyTorch')
        if tf_times:
            ax.bar(x + width, tf_times, width, label='TensorFlow')
        
        ax.set_ylabel('Time (seconds)')
        ax.set_xlabel('Matrix Size')
        ax.set_title('Matrix Multiplication Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{size}x{size}" for size in sizes])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / "matrix_multiplication.png")
        plt.close()
    
    def _plot_inference_performance(self, plots_dir):
        """Plot inference performance comparison"""
        plt.figure(figsize=(10, 6))
        
        labels = ['CNN', 'ResNet50']
        pt_speeds = []
        tf_speeds = []
        
        # Get PyTorch inference speeds
        if "pytorch" in self.results and "inference" in self.results["pytorch"]:
            pt_inf = self.results["pytorch"]["inference"]
            pt_speeds = [
                pt_inf.get("images_per_sec", 0),
                pt_inf.get("resnet50_images_per_sec", 0)
            ]
        
        # Get TensorFlow inference speeds
        if "tensorflow" in self.results and "inference" in self.results["tensorflow"]:
            tf_inf = self.results["tensorflow"]["inference"]
            tf_speeds = [
                tf_inf.get("images_per_sec", 0),
                tf_inf.get("resnet50_images_per_sec", 0)
            ]
        
        # Create the bar chart
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if pt_speeds:
            ax.bar(x - width/2, pt_speeds, width, label='PyTorch')
        if tf_speeds:
            ax.bar(x + width/2, tf_speeds, width, label='TensorFlow')
        
        ax.set_ylabel('Images per second')
        ax.set_xlabel('Model')
        ax.set_title('Inference Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / "inference_performance.png")
        plt.close()
    
    def _plot_io_performance(self, plots_dir):
        """Plot I/O performance"""
        plt.figure(figsize=(10, 6))
        
        # Memory Performance
        if "memory" in self.results:
            mem_res = self.results["memory"]
            sizes = [100, 1000]
            copy_speeds = [mem_res.get(f"copy_speed_{size}mb", 0) for size in sizes]
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(sizes)), copy_speeds)
            plt.xlabel('Array Size (MB)')
            plt.ylabel('Copy Speed (MB/s)')
            plt.title('Memory Copy Performance')
            plt.xticks(range(len(sizes)), sizes)
            plt.tight_layout()
            plt.savefig(plots_dir / "memory_performance.png")
            plt.close()
        
        # Disk Performance
        if "disk" in self.results:
            disk_res = self.results["disk"]
            sizes = [100, 500, 1000]
            read_speeds = [disk_res.get(f"read_speed_{size}mb", 0) for size in sizes]
            write_speeds = [disk_res.get(f"write_speed_{size}mb", 0) for size in sizes]
            
            plt.figure(figsize=(10, 6))
            x = np.arange(len(sizes))
            width = 0.35
            
            plt.bar(x - width/2, read_speeds, width, label='Read')
            plt.bar(x + width/2, write_speeds, width, label='Write')
            
            plt.xlabel('File Size (MB)')
            plt.ylabel('Speed (MB/s)')
            plt.title('Disk I/O Performance')
            plt.xticks(x, sizes)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / "disk_performance.png")
            plt.close()
    
    def _plot_cuda_performance(self, plots_dir):
        """Plot CUDA-specific performance metrics"""
        if "pytorch" not in self.results or "cuda" not in self.results["pytorch"]:
            print("No CUDA performance data available to plot")
            return
        
    # CUDA precision performance comparison
        if "precision" in self.results["pytorch"]["cuda"]:
            precision_data = self.results["pytorch"]["cuda"]["precision"]
        
        # Extract performance data for different precision formats
            formats = []
            tflops = []
        
            for key, value in precision_data.items():
                if key.startswith("matmul_") and key.endswith("_tflops"):
                    format_name = key.replace("matmul_", "").replace("_tflops", "")
                    formats.append(format_name.upper())
                    tflops.append(value)
        
            if formats:
                plt.figure(figsize=(10, 6))
                plt.bar(formats, tflops)
                plt.ylabel('Performance (TFLOPS)')
                plt.xlabel('Precision Format')
                plt.title('CUDA Precision Performance Comparison')
                plt.tight_layout()
                plt.savefig(plots_dir / "cuda_precision.png")
                plt.close()
    
    # Plot memory transfer speeds
        if "transfer" in self.results["pytorch"]["cuda"]:
            transfer_data = self.results["pytorch"]["cuda"]["transfer"]
        
            if "cpu_to_gpu" in transfer_data and "gpu_to_cpu" in transfer_data:
                sizes = []
                cpu_to_gpu = []
                gpu_to_cpu = []
            
                for size, data in transfer_data["cpu_to_gpu"].items():
                    sizes.append(size)
                    cpu_to_gpu.append(data.get("speed_gbps", 0))
            
                for size in sizes:
                    if size in transfer_data["gpu_to_cpu"]:
                        gpu_to_cpu.append(transfer_data["gpu_to_cpu"][size].get("speed_gbps", 0))
                    else:
                        gpu_to_cpu.append(0)
            
                x = np.arange(len(sizes))
                width = 0.35
            
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(x - width/2, cpu_to_gpu, width, label='CPU to GPU')
                ax.bar(x + width/2, gpu_to_cpu, width, label='GPU to CPU')
            
                ax.set_ylabel('Transfer Speed (GB/s)')
                ax.set_xlabel('Matrix Size')
                ax.set_title('CUDA Memory Transfer Performance')
                ax.set_xticks(x)
                ax.set_xticklabels(sizes)
                ax.legend()
            
                plt.tight_layout()
                plt.savefig(plots_dir / "cuda_transfer.png")
                plt.close()

    def save_results(self):
        """Save benchmark results to a JSON file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.output_dir / f"benchmark_results_{timestamp}.json"
    
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {result_file}")
    
        # Save a simple report
        self.report_file = self.output_dir / f"benchmark_report_{timestamp}.txt"
        
    def generate_report(self, no_plots=False):
        """Generate a human-readable report"""
        if not hasattr(self, 'report_file'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.report_file = self.output_dir / f"benchmark_report_{timestamp}.txt"
            
        # Create plots if not disabled
        if not no_plots:
            try:
                self._generate_plots()
            except Exception as e:
                print(f"Error generating plots: {e}")
        else:
            print("Plot generation disabled.")
            
        # Create the text report
        with open(self.report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DEEP LEARNING BENCHMARK REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # System information
            f.write("SYSTEM INFORMATION\n")
            f.write("-" * 80 + "\n")
            
            sys_info = self.results["system_info"]
            f.write(f"Platform: {sys_info.get('platform', 'Unknown')}\n")
            
            if 'cpu_brand' in sys_info:
                f.write(f"CPU: {sys_info['cpu_brand']} ({sys_info['cpu_hz']})\n")
            else:
                f.write(f"CPU: {sys_info.get('processor', 'Unknown')}\n")
                
            f.write(f"CPU Cores: {sys_info.get('physical_cpu_count', 'Unknown')} physical, "
                   f"{sys_info.get('logical_cpu_count', 'Unknown')} logical\n")
            f.write(f"Memory: {sys_info.get('total_memory_gb', 'Unknown')} GB\n")
            
            # GPU information
            f.write("\nGPU INFORMATION\n")
            f.write("-" * 80 + "\n")
            
            if sys_info.get('cuda_available', False):
                f.write(f"CUDA: Available (version {sys_info.get('cuda_version', 'Unknown')})\n")
                f.write(f"CUDA Device: {sys_info.get('cuda_device_name', 'Unknown')}\n")
                
                # Add details for each GPU
                for i in range(sys_info.get('cuda_device_count', 0)):
                    if f"cuda_device_{i}_name" in sys_info:
                        f.write(f"  GPU {i}: {sys_info[f'cuda_device_{i}_name']} "
                              f"({sys_info.get(f'cuda_device_{i}_total_memory', 'Unknown')} GB)\n")
            elif sys_info.get('tf_gpu_available', False):
                f.write("TensorFlow GPU: Available\n")
                for i, gpu_info in enumerate([v for k, v in sys_info.items() if k.startswith('tf_gpu_')]):
                    f.write(f"  GPU {i}: {gpu_info}\n")
            else:
                f.write("GPU: None detected for deep learning\n")
            
            # Performance Summary
            f.write("\nPERFORMANCE SUMMARY\n")
            f.write("-" * 80 + "\n")
            
            # PyTorch Summary
            if "pytorch" in self.results and self.results["pytorch"]:
                pt = self.results["pytorch"]
                f.write("\nPyTorch Performance:\n")
                
                if "inference" in pt and "resnet50_images_per_sec" in pt["inference"]:
                    f.write(f"  ResNet50 inference: {pt['inference'].get('resnet50_images_per_sec', 'N/A')} images/sec\n")
                
                if "inference" in pt and "images_per_sec" in pt["inference"]:
                    f.write(f"  CNN inference: {pt['inference'].get('images_per_sec', 'N/A')} images/sec\n")
                
                if "training" in pt and "cnn_iterations_per_sec" in pt["training"]:
                    f.write(f"  CNN training: {pt['training'].get('cnn_iterations_per_sec', 'N/A')} iterations/sec\n")
                
                if "operations" in pt and "matmul_1024" in pt["operations"]:
                    f.write(f"  Matrix multiplication (1024x1024): {pt['operations'].get('matmul_1024', 'N/A')} seconds\n")
            
            # TensorFlow Summary
            if "tensorflow" in self.results and self.results["tensorflow"]:
                tf_res = self.results["tensorflow"]
                f.write("\nTensorFlow Performance:\n")
                
                if "inference" in tf_res and "resnet50_images_per_sec" in tf_res["inference"]:
                    f.write(f"  ResNet50 inference: {tf_res['inference'].get('resnet50_images_per_sec', 'N/A')} images/sec\n")
                
                if "inference" in tf_res and "images_per_sec" in tf_res["inference"]:
                    f.write(f"  CNN inference: {tf_res['inference'].get('images_per_sec', 'N/A')} images/sec\n")
                
                if "training" in tf_res and "seconds_per_epoch" in tf_res["training"]:
                    f.write(f"  CNN training: {tf_res['training'].get('seconds_per_epoch', 'N/A')} seconds/epoch\n")
                
                if "operations" in tf_res and "matmul_1024" in tf_res["operations"]:
                    f.write(f"  Matrix multiplication (1024x1024): {tf_res['operations'].get('matmul_1024', 'N/A')} seconds\n")
            
            # NumPy Summary
            if "numpy" in self.results and self.results["numpy"]:
                np_res = self.results["numpy"]
                f.write("\nNumPy Performance:\n")
                
                if "matmul_1024" in np_res:
                    f.write(f"  Matrix multiplication (1024x1024): {np_res.get('matmul_1024', 'N/A')} seconds\n")
                
                if "svd_512" in np_res:
                    f.write(f"  SVD decomposition (512x512): {np_res.get('svd_512', 'N/A')} seconds\n")
            
            # I/O Summary
            if "disk" in self.results and self.results["disk"]:
                disk_res = self.results["disk"]
                f.write("\nDisk I/O Performance:\n")
                
                if "read_speed_1000mb" in disk_res:
                    f.write(f"  Read speed: {disk_res.get('read_speed_1000mb', 'N/A')} MB/s\n")
                
                if "write_speed_1000mb" in disk_res:
                    f.write(f"  Write speed: {disk_res.get('write_speed_1000mb', 'N/A')} MB/s\n")
            
            # Memory Summary
            if "memory" in self.results and self.results["memory"]:
                mem_res = self.results["memory"]
                f.write("\nMemory Performance:\n")
                
                if "memory_bandwidth_estimated" in mem_res:
                    f.write(f"  Estimated bandwidth: {mem_res.get('memory_bandwidth_estimated', 'N/A')} MB/s\n")
                
                if "copy_speed_1000mb" in mem_res:
                    f.write(f"  Copy speed (1GB): {mem_res.get('copy_speed_1000mb', 'N/A')} MB/s\n")
            
            # Performance Rating
            f.write("\nOVERALL PERFORMANCE RATING\n")
            f.write("-" * 80 + "\n")
            
            rating = self._calculate_performance_rating()
            f.write(f"Rating: {rating['overall_rating']}/10\n")
            f.write(f"Category: {rating['category']}\n\n")
            f.write(f"Strengths: {', '.join(rating['strengths'])}\n")
            f.write(f"Weaknesses: {', '.join(rating['weaknesses'])}\n\n")
            f.write(f"Recommendation: {rating['recommendation']}\n")
            
            # Footer
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Benchmark completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
            
        print(f"Report saved to: {self.report_file}")
    def benchmark_memory(self):
        """Benchmark memory operations"""
        print("Testing memory performance...")
        results = {}
        
        # Memory allocation and copy speed
        sizes = [10, 100, 1000]  # MB
        for size in sizes:
            size_bytes = size * 1024 * 1024
            
            # Allocation test
            print(f"  Allocating {size} MB array...", end="", flush=True)
            start_time = time.time()
            data = np.zeros(size_bytes, dtype=np.uint8)
            alloc_time = time.time() - start_time
            results[f"alloc_{size}mb"] = alloc_time
            print(f" {alloc_time:.3f} seconds")
            
            # Copy test
            print(f"  Copying {size} MB array...", end="", flush=True)
            target = np.zeros(size_bytes, dtype=np.uint8)
            start_time = time.time()
            np.copyto(target, data)
            copy_time = time.time() - start_time
            results[f"copy_{size}mb"] = copy_time
            results[f"copy_speed_{size}mb"] = size / copy_time  # MB/s
            print(f" {copy_time:.3f} seconds ({size/copy_time:.1f} MB/s)")
        
        # Memory bandwidth - simple estimation
        print("  Estimating memory bandwidth...", end="", flush=True)
        size = 1000  # MB
        size_bytes = size * 1024 * 1024
        iterations = 5
        
        data = np.ones(size_bytes // 8, dtype=np.float64)  # 8 bytes per float64
        
        start_time = time.time()
        for _ in range(iterations):
            result = data * 2.0  # Simple operation to force memory read/write
            
        elapsed = time.time() - start_time
        
        # Each iteration: read data array + write to result array = 2 operations
        # Each operation: size bytes
        # Total bytes: 2 * iterations * size
        bandwidth = (2 * iterations * size) / elapsed  # MB/s
        
        results["memory_bandwidth_estimated"] = bandwidth
        print(f" {elapsed:.3f} seconds ({bandwidth:.1f} MB/s)")
        
        self.results["memory"] = results

    def benchmark_disk_io(self):
        """Benchmark disk I/O operations"""
        print("Testing disk I/O performance...")
        results = {}
        
        # Temporary file location
        test_file = self.output_dir / "benchmark_test_file.bin"
        
        # Write test
        sizes = [100, 500, 1000]  # MB
        for size in sizes:
            size_bytes = size * 1024 * 1024
            data = np.ones(size_bytes, dtype=np.uint8)
            
            # Sequential write
            print(f"  Writing {size} MB file...", end="", flush=True)
            start_time = time.time()
            data.tofile(str(test_file))
            write_time = time.time() - start_time
            write_speed = size / write_time  # MB/s
            
            results[f"write_{size}mb"] = write_time
            results[f"write_speed_{size}mb"] = write_speed
            print(f" {write_time:.3f} seconds ({write_speed:.1f} MB/s)")
            
            # Sequential read
            print(f"  Reading {size} MB file...", end="", flush=True)
            start_time = time.time()
            read_data = np.fromfile(str(test_file), dtype=np.uint8)
            read_time = time.time() - start_time
            read_speed = size / read_time  # MB/s
            
            results[f"read_{size}mb"] = read_time
            results[f"read_speed_{size}mb"] = read_speed
            print(f" {read_time:.3f} seconds ({read_speed:.1f} MB/s)")
        
        # Clean up
        if test_file.exists():
            test_file.unlink()
        
        # Check disk info
        try:
            disk_info = psutil.disk_usage('/')
            results["disk_total"] = disk_info.total / (1024**3)  # GB
            results["disk_used"] = disk_info.used / (1024**3)  # GB
            results["disk_free"] = disk_info.free / (1024**3)  # GB
            results["disk_percent"] = disk_info.percent
        except Exception as e:
            results["disk_info_error"] = str(e)
        
        self.results["disk"] = results    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning System Benchmark")
    parser.add_argument("--output-dir", default="benchmark_results", help="Directory to save results")
    parser.add_argument("--no-plots", action="store_true", help="Disable generation of plots")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("DEEP LEARNING BENCHMARK")
    print("=" * 60)
    print("This script will benchmark your system's performance for deep learning tasks.")
    print("The benchmark will test various components including:")
    print("  - CPU & GPU processing capabilities")
    print("  - Matrix operations performance")
    print("  - Neural network training speed")
    print("  - Inference throughput")
    print("  - Memory and disk performance")
    print("\nThis may take several minutes to complete.")
    print("=" * 60 + "\n")
    
    benchmark = DeepLearningBenchmark(output_dir=args.output_dir, no_plots=args.no_plots)
    benchmark.run_all_benchmarks()