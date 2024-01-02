import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

# Đọc dữ liệu từ file Excel
file_path = 'D:/HOCMAY/winequality-red (1).csv'
df = pd.read_csv(file_path)

# Chọn các đặc trưng cần sử dụng để phân cụm (pH và alcohol)
features = df[['pH', 'alcohol']]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Bắt đầu đo thời gian
start_time = time.time()

# Áp dụng thuật toán DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=3)
clusters = dbscan.fit_predict(scaled_features)
df['Cluster'] = clusters

# Kết thúc đo thời gian và in ra kết quả
end_time = time.time()
execution_time = end_time - start_time
print(f"Thời gian chạy của thuật toán DBSCAN: {execution_time} giây")


# Vẽ biểu đồ phân cụm
plt.scatter(df['pH'], df['alcohol'], c=df['Cluster'], cmap='viridis')
plt.xlabel('pH')
plt.ylabel('Alcohol')
plt.title('DBSCAN Clustering')
plt.show()
