import os
from Get_Data import vn30_list
from Data_Prepareration import StockPricePredictionPipeline

# Test with just one symbol
symbol = vn30_list[0]
print(f"Testing with symbol: {symbol}")

# Create directory for symbol if not exists
save_dir = os.path.join(os.getcwd(), symbol)
os.makedirs(save_dir, exist_ok=True)

pipeline = StockPricePredictionPipeline(
    data_path = "VN30_Full_History_Raw_20251129.csv",
    symbol = symbol,
    window_size = 100,
    save_dir = save_dir
)

# Run complete pipeline with small epochs for speed
pipeline.run_pipeline(epochs=1, batch_size=64)

print(f"Checking if files exist in {save_dir}...")
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
