#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <torch/torch.h>
#include <torch/script.h>

// Add TensorRT headers if available
#ifdef TORCH_TENSORRT_AVAILABLE
#include <torch_tensorrt/torch_tensorrt.h>
#endif

// Constants for the diffusion process
const int IMAGE_SIZE = 32;  // Updated to 32x32 for CIFAR-10
const int CHANNELS = 3;
const int NUM_STEPS = 1000;
const float BETA_START = 0.0001f;
const float BETA_END = 0.02f;

// Function to resize tensor if dimensions don't match
torch::Tensor resize_if_needed(const torch::Tensor& tensor, const c10::IntArrayRef& target_size) {
    if (tensor.size(2) != target_size[2] || tensor.size(3) != target_size[3]) {
        std::cout << "Resizing tensor from [" << tensor.size(0) << ", " << tensor.size(1) << ", " 
                  << tensor.size(2) << ", " << tensor.size(3) << "] to ["
                  << target_size[0] << ", " << target_size[1] << ", " 
                  << target_size[2] << ", " << target_size[3] << "]" << std::endl;
        
        return torch::nn::functional::interpolate(
            tensor,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{target_size[2], target_size[3]})
                .mode(torch::kBilinear)
                .align_corners(false)
        );
    }
    return tensor;
}

class DiffusionSampler {
private:
    torch::jit::script::Module model;
    std::vector<float> betas;
    std::vector<float> alphas;
    std::vector<float> alphas_cumprod;
    std::vector<float> sqrt_alphas_cumprod;
    std::vector<float> sqrt_one_minus_alphas_cumprod;
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen;
    std::normal_distribution<float> normal_dist;

public:
    DiffusionSampler(const std::string& model_path) : gen(rd()), normal_dist(0.0f, 1.0f) {
        try {
            // Check if CUDA is available
            if (torch::cuda::is_available()) {
                std::cout << "CUDA is available! Using GPU." << std::endl;
            } else {
                std::cout << "CUDA is not available. Using CPU." << std::endl;
            }
            
            // Initialize TensorRT runtime (if model path ends with .trt)
            bool is_trt_model = model_path.size() >= 4 && 
                               model_path.substr(model_path.size() - 4) == ".trt";
            
            if (is_trt_model) {
                std::cout << "Loading TensorRT model..." << std::endl;
                
                // Make sure TensorRT operators are registered
                try {
                    // Load the PyTorch model with extra runtime
                    t/                    
                    // For TensorRT models, we need to ensure we're loading to CUDA
                    auto device = torch::kCUDA;
                    model = torch::jit::load(model_path, device, extra_files);
                    model.eval();
                    std::cout << "TensorRT model loaded successfully" << std::endl;
                    
                    // Test the model with a dummy input to ensure it's working
                    try {
                        auto test_input = torch::randn({1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE}, 
                                                      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
                        auto test_timestep = torch::full({1}, 500, 
                                                        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
                        
                        std::vector<torch::jit::IValue> test_inputs;
                        test_inputs.push_back(test_input);
                        test_inputs.push_back(test_timestep);
                        
                        torch::NoGradGuard no_grad;
                        auto test_output = model.forward(test_inputs);
                        std::cout << "Test forward pass successful!" << std::endl;
                        std::cout << "Output shape: " << test_output.toTensor().sizes() << std::endl;
                    } catch (const c10::Error& e) {
                        std::cerr << "Error during test forward pass: " << e.what() << std::endl;
                    }
                } catch (const c10::Error& e) {
                    std::cerr << "Error loading the TensorRT model: " << e.what() << std::endl;
                    std::cerr << "Trying to load as regular model..." << std::endl;
                    
                    // Fallback to regular model loading
                    model = torch::jit::load(model_path);
                    model.eval();
                }
            } else {
                // Load regular PyTorch model
                model = torch::jit::load(model_path);
                model.eval();
                std::cout << "PyTorch model loaded successfully" << std::endl;
            }
        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            throw;
        }
        
        // Initialize diffusion parameters
        initDiffusionParameters();
    }
    
    void initDiffusionParameters() {
        // Linear beta schedule
        betas.resize(NUM_STEPS);
        for (int i = 0; i < NUM_STEPS; i++) {
            float t = static_cast<float>(i) / (NUM_STEPS - 1);
            betas[i] = BETA_START + t * (BETA_END - BETA_START);
        }
        
        // Calculate alphas and cumulative products
        alphas.resize(NUM_STEPS);
        alphas_cumprod.resize(NUM_STEPS);
        sqrt_alphas_cumprod.resize(NUM_STEPS);
        sqrt_one_minus_alphas_cumprod.resize(NUM_STEPS);
        
        float cumprod = 1.0f;
        for (int i = 0; i < NUM_STEPS; i++) {
            alphas[i] = 1.0f - betas[i];
            cumprod *= alphas[i];
            alphas_cumprod[i] = cumprod;
            sqrt_alphas_cumprod[i] = std::sqrt(alphas_cumprod[i]);
            sqrt_one_minus_alphas_cumprod[i] = std::sqrt(1.0f - alphas_cumprod[i]);
        }
    }
    
    torch::Tensor generateNoise(int batch_size) {
        // Generate random noise
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto x = torch::zeros({batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE}, options);
        
        // Fill with random normal values
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < CHANNELS; c++) {
                for (int i = 0; i < IMAGE_SIZE; i++) {
                    for (int j = 0; j < IMAGE_SIZE; j++) {
                        x[b][c][i][j] = normal_dist(gen);
                    }
                }
            }
        }
        
        return x;
    }
    
    torch::Tensor sample(int batch_size) {
        // Start from pure noise
        auto x = generateNoise(batch_size);
        
        // Iteratively denoise
        for (int t = NUM_STEPS - 1; t >= 0; t--) {
            if (t % 100 == 0) {
                std::cout << "Sampling step " << t << "/" << NUM_STEPS << std::endl;
            }
            
            // Add noise for stochastic sampling if t > 0
            float noise_scale = (t > 0) ? std::sqrt(betas[t]) : 0.0f;
            
            // Create timestep tensor (same value for all items in batch)
            auto timestep = torch::full({batch_size}, t, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            
            // Prepare inputs for the model
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(x);
            inputs.push_back(timestep);
            
            // Get model prediction (noise)
            torch::Tensor predicted_noise;
            {
                torch::NoGradGuard no_grad;
                auto output = model.forward(inputs).toTensor();
                predicted_noise = output;
            }
            
            // Resize if dimensions don't match
            predicted_noise = resize_if_needed(predicted_noise, x.sizes());
            
            // Calculate the mean for the reverse process step
            float alpha = alphas[t];
            float alpha_cumprod = alphas_cumprod[t];
            float beta = betas[t];
            
            auto mean_coef1 = 1.0f / std::sqrt(alpha);
            auto mean_coef2 = (1.0f - alpha) / std::sqrt(1.0f - alpha_cumprod);
            
            auto mean = mean_coef1 * (x - mean_coef2 * predicted_noise);
            
            // Add noise if t > 0
            if (t > 0) {
                auto noise = generateNoise(batch_size);
                x = mean + noise_scale * noise;
            } else {
                x = mean;
            }
        }
        
        std::cout << std::endl << "Sampling complete!" << std::endl;
        
        // Normalize to [0, 1] range
        x = (x + 1.0f) / 2.0f;
        x = torch::clamp(x, 0.0f, 1.0f);
        
        return x;
    }
    
    void saveSamples(const torch::Tensor& samples, const std::string& filename) {
        // Convert to CPU if on GPU
        auto cpu_samples = samples.to(torch::kCPU);
        
        // Save using OpenCV or another image library
        // This is a placeholder - you would need to implement actual image saving
        std::cout << "Saving samples to " << filename << std::endl;
        
        // Example using stb_image_write (you would need to include this library)
        // stbi_write_png(filename.c_str(), IMAGE_SIZE, IMAGE_SIZE, CHANNELS, cpu_samples.data_ptr<float>(), IMAGE_SIZE * CHANNELS);
    }
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <batch_size>" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const int batch_size = std::stoi(argv[2]);

    // Check if CUDA is available
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available. This demo requires CUDA." << std::endl;
        return 1;
    }

    try {
        // Create the diffusion sampler
        std::cout << "Initializing diffusion sampler..." << std::endl;
        DiffusionSampler sampler(model_path);
        
        // Generate samples
        std::cout << "Generating samples..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Sample from the model
        torch::Tensor samples = sampler.sample(batch_size);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Sampling completed in " << duration.count() << " ms" << std::endl;
        
        // Convert to CPU and 8-bit format for saving
        torch::Tensor cpu_samples = samples.to(torch::kCPU);
        cpu_samples = (cpu_samples * 255).to(torch::kByte);
        
        std::cout << "Generated " << batch_size << " samples of size " 
                  << cpu_samples.size(2) << "x" << cpu_samples.size(3) << std::endl;
        
        // Here you would typically save the images to disk
        // For simplicity, we'll just print the shape
        std::cout << "Sample tensor shape: " << cpu_samples.sizes() << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
