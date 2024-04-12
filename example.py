from R2CCP.main import R2CCP
import numpy as np
X_train = np.random.rand(1000, 1)
Y_train = 2 * X_train + 1 + 0.1 * np.random.randn(1000, 1)
X_test = np.random.rand(10, 1)
Y_test = 2 * X_test + 1 + 0.1 * np.random.randn(10, 1)

model = R2CCP({'model_path':'model_paths/model_save_destination.pth', 'max_epochs':30})
model.fit(X_train, Y_train)

intervals = model.get_intervals(X_test)
coverage, length = model.get_coverage_length(X_test, Y_test)
predictions = model.predict(X_test)
print(f"Coverage: {np.mean(coverage)}, Length: {np.mean(length)}")