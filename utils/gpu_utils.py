"""
GPU 유틸리티 함수들
"""
import torch
import psutil
import GPUtil

def get_optimal_device():
    """최적의 디바이스를 자동으로 선택"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        return device, 'CUDA'
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        return device, 'MPS'
    else:
        device = torch.device('cpu')
        return device, 'CPU'

def print_device_info(device):
    """디바이스 정보 출력"""
    print(f"\n=== Device Information ===")
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
        # GPU 사용률 정보
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                print(f"GPU Utilization: {gpu.load * 100:.1f}%")
                print(f"GPU Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
        except ImportError:
            print("GPUtil not available - install with: pip install gputil")
            
    elif device.type == 'mps':
        print(f"MPS: Metal Performance Shaders available")
        print(f"PyTorch MPS: {torch.backends.mps.is_built()}")
        
    else:
        print(f"CPU: {psutil.cpu_count()} cores")
        print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    print(f"PyTorch Version: {torch.__version__}")
    print("=" * 30)

def check_gpu_memory(device):
    """GPU 메모리 사용량 체크"""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        cached = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        
        print(f"\nGPU Memory Status:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Cached: {cached:.2f} GB")
        print(f"  Total: {total:.2f} GB")
        print(f"  Free: {total - cached:.2f} GB")
        
        return allocated, cached, total
    else:
        return None, None, None

def optimize_for_device(model, device):
    """디바이스에 최적화된 설정 적용"""
    model = model.to(device)
    
    # MPS 사용 시 특별 설정
    if device.type == 'mps':
        # MPS는 일부 연산에서 float16을 지원하지 않을 수 있음
        print("Using MPS optimizations...")
        # torch.compile은 MPS에서 문제가 있을 수 있으므로 비활성화
        return model
    
    # CUDA 사용 시 최적화
    elif device.type == 'cuda':
        print("Using CUDA optimizations...")
        # Mixed precision 사용 가능
        return model
    
    return model

def get_recommended_batch_size(device, model_size='medium'):
    """디바이스별 권장 배치 사이즈"""
    if device.type == 'cuda':
        # GPU 메모리에 따라 배치 사이즈 조정
        try:
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
            if total_memory >= 16:
                return 32 if model_size == 'small' else 16
            elif total_memory >= 8:
                return 16 if model_size == 'small' else 8
            else:
                return 8 if model_size == 'small' else 4
        except:
            return 8
            
    elif device.type == 'mps':
        # MPS는 메모리 관리가 다르므로 보수적으로 설정
        return 8 if model_size == 'small' else 4
    
    else:
        # CPU는 메모리 사용량이 다르므로 더 작은 배치 사이즈
        return 4 if model_size == 'small' else 2
