import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Mock vn30_list
vn30_list = ["ACB"]

class MockPipeline:
    def __init__(self, data_path, symbol, window_size=100, save_dir='.'):
        self.symbol = symbol
        self.save_dir = save_dir
        self.data = pd.DataFrame({'Close': np.random.rand(100)})
        self.model = "MockModel"

    def visualize_data(self):
        print("Mocking visualize_data...")
        plt.figure()
        plt.plot(self.data['Close'])
        plt.savefig(os.path.join(self.save_dir, f'{self.symbol}_data_overview.png'))
        plt.close()
        
        plt.figure()
        plt.plot(self.data['Close'])
        plt.savefig(os.path.join(self.save_dir, f'{self.symbol}_trading_data.png'))
        plt.close()
        
        plt.figure()
        plt.plot(self.data['Close'])
        plt.savefig(os.path.join(self.save_dir, f'{self.symbol}_close_price.png'))
        plt.close()

    def visualize_training(self):
        print("Mocking visualize_training...")
        plt.figure()
        plt.plot([1, 2, 3])
        plt.savefig(os.path.join(self.save_dir, f'{self.symbol}_training_history.png'))
        plt.close()

    def visualize_predictions(self, predicted, test_label):
        print("Mocking visualize_predictions...")
        plt.figure()
        plt.plot(predicted)
        plt.savefig(os.path.join(self.save_dir, f'{self.symbol}_predictions.png'))
        plt.close()

    def save_model_architecture(self, filename='model.png'):
        print("Mocking save_model_architecture...")
        # Just save a dummy file
        with open(os.path.join(self.save_dir, filename), 'w') as f:
            f.write("Mock model architecture")

    def run_pipeline(self, epochs=100, batch_size=64):
        self.visualize_data()
        self.visualize_training()
        self.visualize_predictions([1, 2, 3], [1, 2, 3])
        self.save_model_architecture()

# Test logic from main.py
print("Testing main.py logic...")
for symbol in vn30_list:
    # Create directory for symbol if not exists
    save_dir = os.path.join(os.getcwd(), symbol)
    os.makedirs(save_dir, exist_ok=True)

    pipeline = MockPipeline(
        data_path = "VN30_Full_History_Raw_20251129.csv",
        symbol = symbol,
        window_size = 100,
        save_dir = save_dir
    )

    # Run complete pipeline
    pipeline.run_pipeline(epochs=1, batch_size=64)

    print(f"Checking if files exist...")
    expected_files = [
        f'{symbol}_data_overview.png',
        f'{symbol}_trading_data.png',
        f'{symbol}_close_price.png',
        f'{symbol}_training_history.png',
        f'{symbol}_predictions.png',
        'model.png'
    ]

    all_exist = True
    for f in expected_files:
        path = os.path.join(save_dir, f)
        if os.path.exists(path):
            print(f"[OK] {f} exists.")
        else:
            print(f"[FAIL] {f} does not exist.")
            all_exist = False

    if all_exist:
        print("Verification SUCCESS!")
    else:
        print("Verification FAILED!")
