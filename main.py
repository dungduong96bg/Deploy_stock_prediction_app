import os
from Get_Data import get_full_vn30_history, get_data_version, vn30_list
from Data_Prepareration import StockPricePredictionPipeline


if __name__ == '__main__':
    data = get_data_version(crawl = False)

    # Initialize pipeline
    for symbol in vn30_list:
        # Create directory for symbol if not exists
        save_dir = os.path.join(os.getcwd(), symbol)
        os.makedirs(save_dir, exist_ok=True)

        pipeline = StockPricePredictionPipeline(
            data_path = "VN30_Full_History_Raw_20251129.csv",
            symbol = symbol,
            window_size = 100,
            save_dir = save_dir
        )

        # Run complete pipeline
        pipeline.run_pipeline(epochs=100, batch_size=64)

