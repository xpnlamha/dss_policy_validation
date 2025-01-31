import numpy as np

def evaluate_model_performance(true_labels, predictions, n_bootstrap=1000):
    n_samples = len(true_labels)
    bootstrap_accuracies = []
    
    # Thực hiện bootstrap sampling
    for _ in range(n_bootstrap):
        # Lấy mẫu ngẫu nhiên có phục hồi
        indices = np.random.randint(0, n_samples, size=n_samples)
        sample_true = true_labels[indices]
        sample_pred = predictions[indices]
        
        # Tính accuracy cho mẫu bootstrap
        accuracy = np.mean(sample_true == sample_pred)
        bootstrap_accuracies.append(accuracy)
    
    # Tính các metric đánh giá
    mean_accuracy = np.mean(bootstrap_accuracies)
    std_accuracy = np.std(bootstrap_accuracies)
    se = std_accuracy / np.sqrt(n_bootstrap)
    
    # Tính khoảng tin cậy 95%
    ci_lower = mean_accuracy - 1.96 * se
    ci_upper = mean_accuracy + 1.96 * se
    
    # Tính relative standard error
    relative_se = se / mean_accuracy
    
    return {
        'mean_accuracy': mean_accuracy,
        'standard_error': se,
        'confidence_interval': (ci_lower, ci_upper),
        'relative_standard_error': relative_se
    }

if __name__ == "__main__":
    # Tạo dữ liệu mẫu
    n_samples = 1000
    
    # Tạo true labels (ground truth)
    true_labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 100)  # 1: spam, 0: không spam
    
    # Tạo predictions từ mô hình
    model_predictions = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0] * 100)
    
    # Tính accuracy ban đầu
    initial_accuracy = np.mean(true_labels == model_predictions)
    print(f"Accuracy ban đầu: {initial_accuracy:.3f}")
    
    # Thực hiện đánh giá
    results = evaluate_model_performance(true_labels, model_predictions)
    
    print(f"\nKết quả Performance Metric Evaluation:")
    print(f"Accuracy trung bình: {results['mean_accuracy']:.3f}")
    print(f"Standard Error: {results['standard_error']:.8f}")
    print(f"Khoảng tin cậy 95%: ({results['confidence_interval'][0]:.8f}, {results['confidence_interval'][1]:.8f})")
    print(f"Relative Standard Error: {results['relative_standard_error']:.8f}")
    
    print("\nSo sánh kết quả:")
    print(f"Accuracy ban đầu: {initial_accuracy}")
    print(f"Accuracy từ Performance Metric Evaluation: {results['mean_accuracy']:.8f}")
    print(f"Độ lệch: {abs(initial_accuracy - results['mean_accuracy']):.8f}")  