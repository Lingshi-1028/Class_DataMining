# ----------------------------------------------------------------
# ---------------------- 电影评价数据协同过滤 ----------------------
# ----------------------------------------------------------------
import pandas as pd
from surprise import Dataset, Reader, KNNBasic, SVD
from surprise.model_selection import cross_validate
import os

def main():
    # 1. 配置文件路径（此处根据自己的文件目录进行配置）
    # 根目录：Exercise_2
    root_dir = r"C:\Users\Lenovo\Desktop\数据挖掘\作业提交\Exercise_2"
    # 数据目录：Exercise_2\data（存放4个.csv文件）
    data_dir = os.path.join(root_dir, "data")
    # 核心评分数据文件路径
    ratings_file_path = os.path.join(data_dir, "ratings.csv")
    
    # 2. 数据加载配置
    reader = Reader(rating_scale=(0.5, 5.0))  # 定义评分范围（适配常规评分数据）
    
    # 加载指定路径下的csv文件
    ratings_df = pd.read_csv(ratings_file_path)
    # 转换为Surprise兼容格式（需确保csv包含userId, movieId, rating三列，符合常规数据格式）
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    
    # 3. 初始化协同过滤算法
    algo = SVD()  # 稳定高效的矩阵分解算法
    
    # 4. 10折交叉验证，评估RMSE和MAE指标
    cv_results = cross_validate(
        algo, data, 
        measures=['RMSE', 'MAE'],
        cv=10,
        verbose=True
    )
    
    # 5. 输出最终平均结果
    print("\n10折交叉验证最终结果：")
    print(f"平均RMSE: {cv_results['test_rmse'].mean():.4f}")
    print(f"平均MAE: {cv_results['test_mae'].mean():.4f}")

if __name__ == "__main__":

    main()





    