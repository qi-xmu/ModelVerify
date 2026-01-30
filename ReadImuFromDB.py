#!/usr/bin/env python3
"""
从数据库中读取IMU数据并保存为imu.csv文件
"""

import sqlite3
import os
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Optional

class IMUDataReader:
    """从数据库读取IMU数据的类"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        
    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def connect(self):
        """连接到数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"成功连接到数据库: {self.db_path}")
        except sqlite3.Error as e:
            print(f"连接数据库失败: {e}")
            raise
            
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            
    def check_imu_table(self) -> bool:
        """检查是否存在imu_data表"""
        if not self.conn:
            raise sqlite3.Error("数据库未连接")
            
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='imu_data'"
        )
        result = cursor.fetchone()
        return result is not None
        
    def read_imu_data(self) -> pd.DataFrame:
        """从数据库读取IMU数据"""
        if not self.conn:
            raise sqlite3.Error("数据库未连接")
            
        if not self.check_imu_table():
            raise sqlite3.Error("数据库中不存在imu_data表")
            
        query = """
        SELECT 
            timestamp_ns,
            w_RS_s_x, w_RS_s_y, w_RS_s_z,
            a_RS_s_x, a_RS_s_y, a_RS_s_z,
            q_RS_w, q_RS_x, q_RS_y, q_RS_z,
            t_system_ns,
            m_RS_s_x, m_RS_s_y, m_RS_s_z
        FROM imu_data 
        ORDER BY timestamp_ns
        """
        
        try:
            df = pd.read_sql_query(query, self.conn)
            
            # 重命名列以匹配CSV格式
            column_mapping = {
                'timestamp_ns': 'timestamp',
                'w_RS_s_x': 'w_RS_S_x',
                'w_RS_s_y': 'w_RS_S_y', 
                'w_RS_s_z': 'w_RS_S_z',
                'a_RS_s_x': 'a_RS_S_x',
                'a_RS_s_y': 'a_RS_S_y',
                'a_RS_s_z': 'a_RS_S_z',
                'q_RS_w': 'q_RS_w',
                'q_RS_x': 'q_RS_x',
                'q_RS_y': 'q_RS_y',
                'q_RS_z': 'q_RS_z',
                't_system_ns': 't_system',
                'm_RS_s_x': 'm_RS_S_x',
                'm_RS_s_y': 'm_RS_S_y',
                'm_RS_s_z': 'm_RS_S_z'
            }
            
            df = df.rename(columns=column_mapping)
            
            # 将纳秒时间戳转换为微秒
            # 选项1: 保持整数格式
            # df['timestamp'] = (df['timestamp'] / 1000).astype('int64')  # ns -> μs
            # df['t_system'] = (df['t_system'] / 1000).astype('int64')  # ns -> μs
            
            # 选项2: 保持浮点数格式
            df['timestamp'] = df['timestamp'] / 1000  # ns -> μs
            df['t_system'] = df['t_system'] / 1000  # ns -> μs
            
            # 注释掉浮点数精度控制代码，保持原始数据的精度
            # float_columns = ['w_RS_S_x', 'w_RS_S_y', 'w_RS_S_z', 
            #                'a_RS_S_x', 'a_RS_S_y', 'a_RS_S_z',
            #                'q_RS_w', 'q_RS_x', 'q_RS_y', 'q_RS_z',
            #                'm_RS_S_x', 'm_RS_S_y', 'm_RS_S_z']
            # 
            # for col in float_columns:
            #     df[col] = df[col].astype(float)
            
            print(f"成功读取 {len(df)} 条IMU数据记录")
            return df
            
        except sqlite3.Error as e:
            print(f"读取IMU数据失败: {e}")
            raise
            
    def save_to_csv(self, df: pd.DataFrame, output_path: str):
        """将数据保存为CSV文件"""
        try:
            # 添加列标题注释行（包含单位信息）
            header_line = "#timestamp [us],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],t_system [us],m_RS_S_x [µT],m_RS_S_y [µT],m_RS_S_z [µT]"
            
            # 直接写入文件，先写标题行，再写数据（不包含列名，保持原始精度）
            with open(output_path, 'w') as f:
                f.write(header_line + '\n')
                # 写入数据，不包含列名，保持原始精度，不控制浮点数格式
                df.to_csv(f, index=False, header=False)
                
            print(f"IMU数据已保存到: {output_path}")
            print(f"保存了 {len(df)} 条记录")
            
        except Exception as e:
            print(f"保存CSV文件失败: {e}")
            raise

def find_dataset_databases(data_dir: str = "data") -> List[str]:
    """查找所有dataset.db文件"""
    db_files = []
    data_path = Path(data_dir)
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file == 'dataset.db':
                db_files.append(os.path.join(root, file))
    
    return db_files
def find_dataset_units(data_dir: str = "data") -> List[str]:
    """查找所有包含dataset.db的数据单元文件夹"""
    unit_dirs = set()
    data_path = Path(data_dir)
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file == 'dataset.db':
                unit_dirs.add(root)  # 将dataset.db所在的目录作为一个数据单元
    
    return sorted(list(unit_dirs))

def process_unit_directory(unit_dir: str):
    """处理单个数据单元（包含dataset.db的目录）"""
    if not os.path.exists(unit_dir):
        print(f"数据单元目录不存在: {unit_dir}")
        return
    
    db_path = os.path.join(unit_dir, 'dataset.db')
    if not os.path.exists(db_path):
        print(f"数据单元 {unit_dir} 中没有找到dataset.db文件")
        return
    
    output_path = os.path.join(unit_dir, 'imu_from_db.csv')
    print(f"处理数据单元: {unit_dir}")
    process_single_database(db_path, output_path)

def process_batch_units(data_dir: str = "data"):
    """批处理数据单元目录（每个包含dataset.db的子目录作为一个单元）"""
    unit_dirs = find_dataset_units(data_dir)
    
    if not unit_dirs:
        print(f"在 {data_dir} 目录中没有找到包含dataset.db的数据单元")
        return
    
    print(f"找到 {len(unit_dirs)} 个数据单元")
    
    for unit_dir in unit_dirs:
        process_unit_directory(unit_dir)
def process_single_database(db_path: str, output_path: Optional[str] = None):
    """处理单个数据库文件"""
    if not os.path.exists(db_path):
        print(f"数据库文件不存在: {db_path}")
        return
        
    # 如果没有指定输出路径，使用数据库所在目录
    if output_path is None:
        output_path = os.path.join(os.path.dirname(db_path), 'imu_from_db.csv')
    
    try:
        with IMUDataReader(db_path) as reader:
            if reader.check_imu_table():
                df = reader.read_imu_data()
                reader.save_to_csv(df, output_path)
            else:
                print(f"数据库 {db_path} 中没有imu_data表")
                
    except Exception as e:
        print(f"处理数据库 {db_path} 时出错: {e}")

def process_all_databases(data_dir: str = "data"):
    """处理所有找到的数据库文件"""
    db_files = find_dataset_databases(data_dir)
    
    if not db_files:
        print(f"在 {data_dir} 目录中没有找到dataset.db文件")
        return
    
    print(f"找到 {len(db_files)} 个dataset.db文件")
    
    for db_path in db_files:
        print(f"\n处理: {db_path}")
        output_path = os.path.join(os.path.dirname(db_path), 'imu_from_db.csv')
        process_single_database(db_path, output_path)

def main():
    parser = argparse.ArgumentParser(description="从数据库读取IMU数据并保存为CSV文件")
    parser.add_argument("--db", type=str, help="指定单个数据库文件路径")
    parser.add_argument("--output", type=str, help="指定输出CSV文件路径")
    parser.add_argument("--all", action="store_true", help="处理所有找到的数据库文件")
    parser.add_argument("--data-dir", type=str, default="data", help="数据目录路径")
    parser.add_argument("-u", "--unit", type=str, help="指定单个数据单元目录（包含dataset.db的文件夹）")
    parser.add_argument("-d", "--dir", type=str, help="指定批处理的上级目录，将处理每个包含dataset.db的子目录作为一个数据单元")
    
    args = parser.parse_args()
    
    if args.db:
        # 处理单个数据库
        process_single_database(args.db, args.output)
    elif args.unit:
        # 处理单个数据单元
        process_unit_directory(args.unit)
    elif args.dir:
        # 批处理数据单元
        process_batch_units(args.dir)
    elif args.all:
        # 处理所有数据库（原有功能）
        process_all_databases(args.data_dir)
    else:
        print("使用方式:")
        print("  --db 参数处理单个数据库文件")
        print("  -u/--unit 参数处理单个数据单元目录")
        print("  -d/--dir 参数批处理数据单元目录")
        print("  --all 参数处理所有找到的数据库文件")
        print("使用 --help 查看详细帮助")

if __name__ == "__main__":
    main()