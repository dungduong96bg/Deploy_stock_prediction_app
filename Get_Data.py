import requests
import pandas as pd
from datetime import datetime
import time
import json

vn30_list = [
        "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
        "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
        "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
    ]

def get_stock_data(symbol, start_date="2000-01-01", end_date=None):
    """
    Fetch historical data for a single stock symbol from FireAnt.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    base_url = "https://svr2.fireant.vn/api/Data/Companies/HistoricalQuotes"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Referer': 'https://fireant.vn/',
        'Origin': 'https://fireant.vn'
    }

    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")

    try:
        params = {
            'symbol': symbol,
            'startDate': start_date,
            'endDate': end_date
        }

        response = requests.get(base_url, params=params, headers=headers, timeout=15)

        if response.status_code == 200:
            data = response.json()

            if data and isinstance(data, list) and len(data) > 0:
                df_temp = pd.DataFrame(data)
                df_temp['Symbol'] = symbol
                if 'Date' in df_temp.columns:
                    df_temp['Date'] = pd.to_datetime(df_temp['Date']).dt.date
                
                # Rename columns to match expected format
                rename_map = {
                    'AdjClose': 'Close', 
                    'AdjOpen': 'Open', 
                    'AdjHigh': 'High', 
                    'AdjLow': 'Low'
                }
                # Only rename if columns exist
                df_temp = df_temp.rename(columns={k: v for k, v in rename_map.items() if k in df_temp.columns})
                
                # Ensure required columns exist
                required_cols = ['Date', 'Symbol', 'Close', 'Open', 'High', 'Low', 'Volume']
                available_cols = [col for col in required_cols if col in df_temp.columns]
                df_temp = df_temp[available_cols]
                
                print(f"Successfully fetched {len(df_temp)} records for {symbol}.")
                return df_temp
            else:
                print(f"No data returned for {symbol}.")
                return None
        else:
            print(f"HTTP Error {response.status_code} for {symbol}.")
            return None

    except Exception as e:
        print(f"Exception fetching data for {symbol}: {e}")
        return None


def get_full_vn30_history():
    # 1. Cấu hình thời gian
    start_date = "2000-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    all_data_frames = []

    print(f"--- BẮT ĐẦU CRAWL DỮ LIỆU VN30 ---")
    print(f"Khoảng thời gian: {start_date} đến {end_date}")

    for symbol in vn30_list:
        df = get_stock_data(symbol, start_date, end_date)
        if df is not None:
            all_data_frames.append(df)
        
        # Delay để tránh bị chặn IP
        time.sleep(1.5)

    # 4. Gộp và Lưu dữ liệu
    if all_data_frames:
        print("\nĐang tổng hợp dữ liệu...")
        final_df = pd.concat(all_data_frames, ignore_index=True)
        
        # Đặt tên file có timestamp
        file_name = f"VN30_Full_History_Raw_{datetime.now().strftime('%Y%m%d')}.csv"

        # Lưu file CSV
        final_df.to_csv(file_name, index=False, encoding='utf-8-sig')

        print("-" * 50)
        print(f"HOÀN TẤT!")
        print(f"Tổng số dòng: {len(final_df)}")
        print(f"File đã lưu: {file_name}")

        return final_df
    else:
        print("Không thu thập được dữ liệu nào.")
        return None

def get_data_version(crawl = False):
    if crawl:
        return get_full_vn30_history()
    else:
        return pd.read_csv('VN30_Full_History_Raw_20251129.csv')

if __name__ == "__main__":
    # Chạy hàm
    df = get_full_vn30_history()
