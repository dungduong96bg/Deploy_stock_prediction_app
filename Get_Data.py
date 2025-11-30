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

def get_full_vn30_history():
    # 1. Cấu hình thời gian
    # Lấy từ quá khứ xa (năm 2000) để đảm bảo lấy hết dữ liệu lịch sử
    start_date = "2000-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")  # Lấy đến hiện tại (2025-11-29)

    # 3. Cấu hình API
    base_url = "https://svr2.fireant.vn/api/Data/Companies/HistoricalQuotes"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Referer': 'https://fireant.vn/',
        'Origin': 'https://fireant.vn'
    }

    all_data_frames = []

    print(f"--- BẮT ĐẦU CRAWL DỮ LIỆU VN30 ---")
    print(f"Khoảng thời gian: {start_date} đến {end_date}")

    for symbol in vn30_list:
        print(f"Đang xử lý: {symbol}...", end=" ")

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
                    # Chuyển đổi list of dictionaries thành DataFrame
                    # Pandas sẽ tự động tạo cột cho TẤT CẢ các key có trong JSON (bao gồm AdjRatio, PE, PB...)
                    df_temp = pd.DataFrame(data)

                    # Đảm bảo cột Symbol chính xác (đề phòng API trả về null ở cột này)
                    df_temp['Symbol'] = symbol

                    # Xử lý format ngày tháng cho chuẩn Excel/CSV
                    if 'Date' in df_temp.columns:
                        df_temp['Date'] = pd.to_datetime(df_temp['Date']).dt.date

                    all_data_frames.append(df_temp)
                    print(f"OK! Lấy được {len(df_temp)} bản ghi.")
                else:
                    print("Không có dữ liệu trả về.")
            else:
                print(f"Lỗi HTTP: {response.status_code}")

        except Exception as e:
            print(f"Lỗi ngoại lệ: {e}")

        # Delay để tránh bị chặn IP
        time.sleep(1.5)

    # 4. Gộp và Lưu dữ liệu
    if all_data_frames:
        print("\nĐang tổng hợp dữ liệu...")
        final_df = pd.concat(all_data_frames, ignore_index=True)

        final_df = final_df[['Date', 'Symbol' , 'AdjClose', 'AdjOpen', 'AdjHigh', 'AdjLow','Volume']]\
            .rename(columns = {'AdjClose' : 'Close', 'AdjOpen' : 'Open', 'AdjHigh' : 'High', 'AdjLow' : 'Low'})
        # Đặt tên file có timestamp
        file_name = f"VN30_Full_History_Raw_{datetime.now().strftime('%Y%m%d')}.csv"

        # Lưu file CSV với encoding utf-8-sig để đọc được tiếng Việt và ký tự đặc biệt trên Excel
        final_df.to_csv(file_name, index=False, encoding='utf-8-sig')

        print("-" * 50)
        print(f"HOÀN TẤT!")
        print(f"Tổng số dòng: {len(final_df)}")
        print(f"Số lượng cột (fields): {len(final_df.columns)}")
        print(f"Các cột đã crawl: {list(final_df.columns)}")
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
