#!/usr/bin/env python3
"""
将gt.csv文件中的真值数据写入数据库

# 处理单个文件夹
uv run WriteGtToDB.py -u /path/to/folder/

# 批处理整个数据目录
uv run WriteGtToDB.py -d data

# 详细模式显示表结构和数据
uv run WriteGtToDB.py -u data/20260126_174027_Redmi-K30-Pro -v

# 批处理并显示详细信息
uv run WriteGtToDB.py -d data -v
"""

import sqlite3
import os
import sys
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Optional

class GTDataWriter:
    """将GT数据写入数据库的类"""
    
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
            
    def create_gt_table(self):
        """创建gt_data表"""
        if not self.conn:
            raise sqlite3.Error("数据库未连接")
            
        cursor = self.conn.cursor()
        
        # 删除已存在的表（如果需要）
        cursor.execute("DROP TABLE IF EXISTS gt_data")
        self.conn.commit()
        
        # 创建新表
        create_table_query = """
        CREATE TABLE gt_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_us REAL NOT NULL,
            p_RS_R_x REAL NOT NULL,
            p_RS_R_y REAL NOT NULL,
            p_RS_R_z REAL NOT NULL,
            q_RS_w REAL NOT NULL,
            q_RS_x REAL NOT NULL,
            q_RS_y REAL NOT NULL,
            q_RS_z REAL NOT NULL
        )
        """
        
        cursor.execute(create_table_query)
        self.conn.commit()
        
        # 验证表是否正确创建
        cursor.execute("PRAGMA table_info(gt_data)")
        columns = cursor.fetchall()
        print("成功创建gt_data表，表结构:")
        for col in columns:
            print(f"  {col[1]} {col[2]}")
        
    def read_gt_csv(self, csv_path: str) -> pd.DataFrame:
        """从CSV文件读取GT数据"""
        try:
            # 先读取第一行，判断是否为注释行
            with open(csv_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('#'):
                    # 如果第一行是注释，则使用下一行作为列名
                    df = pd.read_csv(csv_path, skiprows=1)
                    # 手动设置列名
                    df.columns = [
                        'timestamp [us]', 'p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]',
                        'q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []'
                    ]
                else:
                    # 否则直接读取
                    df = pd.read_csv(csv_path, comment='#')
            
            # 确保列名正确
            expected_columns = [
                'timestamp [us]', 'p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]',
                'q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []'
            ]
            
            # 检查是否有单位列名
            if all(col in df.columns for col in expected_columns):
                # 重命名列以去掉单位信息
                df = df.rename(columns={
                    'timestamp [us]': 'timestamp',
                    'p_RS_R_x [m]': 'p_RS_R_x',
                    'p_RS_R_y [m]': 'p_RS_R_y',
                    'p_RS_R_z [m]': 'p_RS_R_z',
                    'q_RS_w []': 'q_RS_w',
                    'q_RS_x []': 'q_RS_x',
                    'q_RS_y []': 'q_RS_y',
                    'q_RS_z []': 'q_RS_z'
                })
            
            # 检查数据完整性
            required_columns = ['timestamp', 'p_RS_R_x', 'p_RS_R_y', 'p_RS_R_z',
                               'q_RS_w', 'q_RS_x', 'q_RS_y', 'q_RS_z']
            
            if not all(col in df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"CSV文件缺少必需的列: {missing_cols}\n实际列名: {list(df.columns)}")
                
            print(f"成功读取 {len(df)} 条GT数据记录")
            return df
            
        except Exception as e:
            print(f"读取CSV文件失败: {e}")
            raise
            
    def write_gt_data(self, df: pd.DataFrame):
        """将GT数据写入数据库"""
        if not self.conn:
            raise sqlite3.Error("数据库未连接")
            
        cursor = self.conn.cursor()
        
        # 准备插入语句
        insert_query = """
        INSERT INTO gt_data (timestamp_us, p_RS_R_x, p_RS_R_y, p_RS_R_z, 
                            q_RS_w, q_RS_x, q_RS_y, q_RS_z)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            # 准备数据，确保列名匹配
            # 重命名timestamp列为timestamp_us以匹配数据库
            df_insert = df.rename(columns={'timestamp': 'timestamp_us'})
            data_to_insert = df_insert[[
                'timestamp_us', 'p_RS_R_x', 'p_RS_R_y', 'p_RS_R_z',
                'q_RS_w', 'q_RS_x', 'q_RS_y', 'q_RS_z'
            ]].values.tolist()
            
            # 准备数据
            
            # 批量插入数据
            cursor.executemany(insert_query, data_to_insert)
            self.conn.commit()
            
            print(f"成功写入 {len(data_to_insert)} 条GT数据记录到数据库")
            
        except sqlite3.Error as e:
            print(f"写入GT数据失败: {e}")
            self.conn.rollback()
            raise
            
    def verify_data(self) -> bool:
        """验证写入的数据"""
        if not self.conn:
            raise sqlite3.Error("数据库未连接")
            
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM gt_data")
        count = cursor.fetchone()[0]
        
        if count > 0:
            print(f"验证成功: gt_data表中有 {count} 条记录")
            
            # 显示前几条记录作为示例
            cursor.execute("SELECT * FROM gt_data LIMIT 5")
            records = cursor.fetchall()
            print("前5条记录:")
            for record in records:
                print(f"  {record}")
            return True
        else:
            print("验证失败: gt_data表为空")
            return False

def process_single_csv(csv_path: str, db_path: str, verbose: bool = False):
    """处理单个CSV文件"""
    if not os.path.exists(csv_path):
        print(f"CSV文件不存在: {csv_path}")
        return
        
    # 如果数据库文件不存在，创建它
    if not os.path.exists(db_path):
        print(f"数据库文件不存在，将创建: {db_path}")
        
    try:
        print(f"开始处理: {csv_path} -> {db_path}")
        with GTDataWriter(db_path) as writer:
            # 创建表
            writer.create_gt_table()
            
            # 读取数据
            df = writer.read_gt_csv(csv_path)
            print(f"读取到 {len(df)} 条记录")
            
            # 写入数据
            writer.write_gt_data(df)
            
            # 验证数据
            writer.verify_data()
            
            # 如果需要详细信息，验证数据库表
            if verbose:
                verify_database_tables(db_path, verbose)
                
    except Exception as e:
        print(f"处理CSV文件 {csv_path} 时出错: {e}")

def find_gt_csv_files(data_dir: str = "data") -> List[Dict[str, str]]:
    """查找所有gt.csv文件并确定对应的数据库路径"""
    csv_files = []
    data_path = Path(data_dir)
    
    for root, dirs, files in os.walk(data_path):
        if 'gt.csv' in files:
            csv_path = os.path.join(root, 'gt.csv')
            # 对应的数据库文件在同一目录下
            db_path = os.path.join(root, 'dataset.db')
            csv_files.append({
                'csv_path': csv_path,
                'db_path': db_path,
                'folder': root
            })
    
    return csv_files

def process_batch_folders(data_dir: str = "data", verbose: bool = False):
    """批处理所有包含gt.csv和dataset.db的文件夹"""
    csv_files = find_gt_csv_files(data_dir)
    
    if not csv_files:
        print(f"在 {data_dir} 目录中没有找到gt.csv文件")
        return
    
    print(f"找到 {len(csv_files)} 个包含gt.csv文件的文件夹:")
    
    # 按上层文件夹分组
    folders_by_parent = {}
    for file_info in csv_files:
        folder = Path(file_info['folder'])
        parent_folder = folder.parent
        if parent_folder not in folders_by_parent:
            folders_by_parent[parent_folder] = []
        folders_by_parent[parent_folder].append(file_info)
    
    print("\n按上层文件夹分组:")
    for parent_folder, files in folders_by_parent.items():
        print(f"\n上层文件夹: {parent_folder}")
        for file_info in files:
            print(f"  - {file_info['folder']}")
    
    # 处理每个文件夹
    for parent_folder, files in folders_by_parent.items():
        print(f"\n===== 处理上层文件夹: {parent_folder} =====")
        
        for file_info in files:
            folder = file_info['folder']
            csv_path = file_info['csv_path']
            db_path = file_info['db_path']
            
            print(f"\n处理文件夹: {folder}")
            process_single_csv(csv_path, db_path, verbose)

def verify_database_tables(db_path: str, verbose: bool = False):
    """验证数据库表结构，检查gt表是否新增"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"\n数据库 {db_path} 中的表:")
        for table in tables:
            print(f"  - {table}")
        
        # 检查gt表是否存在
        if 'gt' in tables:
            print(f"\n✓ gt表已存在")
            
            # 显示gt表的结构
            cursor.execute("PRAGMA table_info(gt)")
            columns = cursor.fetchall()
            print(f"\ngt表结构:")
            for col in columns:
                print(f"  {col[1]} {col[2]} ({'NOT NULL' if col[3] == 1 else 'NULL'})")
            
            # 显示gt表的前5行数据
            if verbose:
                cursor.execute("SELECT * FROM gt LIMIT 5")
                rows = cursor.fetchall()
                print(f"\ngt表前5行数据:")
                for row in rows:
                    print(f"  {row}")
        
        elif 'gt_data' in tables:
            print(f"\n✓ gt_data表已存在")
            
            # 显示gt_data表的结构
            cursor.execute("PRAGMA table_info(gt_data)")
            columns = cursor.fetchall()
            print(f"\ngt_data表结构:")
            for col in columns:
                print(f"  {col[1]} {col[2]} ({'NOT NULL' if col[3] == 1 else 'NULL'})")
            
            # 显示gt_data表的前5行数据
            if verbose:
                # 检查表是否有数据
                cursor.execute("SELECT COUNT(*) FROM gt_data")
                count = cursor.fetchone()[0]
                if count > 0:
                    cursor.execute("SELECT * FROM gt_data LIMIT 5")
                    rows = cursor.fetchall()
                    print(f"\ngt_data表前5行数据:")
                    for row in rows:
                        print(f"  {row}")
                else:
                    print(f"\ngt_data表无数据")
        else:
            print(f"\n✗ 未找到gt或gt_data表")
        
        conn.close()
        
    except Exception as e:
        print(f"验证数据库时出错: {e}")

def process_all_csv_files(data_dir: str = "data", verbose: bool = False):
    """处理所有找到的CSV文件"""
    csv_files = find_gt_csv_files(data_dir)
    
    if not csv_files:
        print(f"在 {data_dir} 目录中没有找到gt.csv文件")
        return
    
    print(f"找到 {len(csv_files)} 个gt.csv文件")
    
    for file_info in csv_files:
        print(f"\n处理: {file_info['csv_path']}")
        process_single_csv(file_info['csv_path'], file_info['db_path'], verbose)

def main():
    parser = argparse.ArgumentParser(description="将gt.csv文件中的真值数据写入数据库")
    parser.add_argument("-u", "--folder", type=str, help="包含gt.csv和dataset.db文件的文件夹路径")
    parser.add_argument("-d", "--data-dir", type=str, default="data", help="批处理的数据目录路径")
    parser.add_argument("-csv", type=str, help="指定单个GT CSV文件路径")
    parser.add_argument("-db", type=str, help="指定数据库文件路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细信息模式，显示数据库表结构和数据")
    parser.add_argument("--all", action="store_true", help="处理所有找到的GT CSV文件")
    
    args = parser.parse_args()
    
    if args.csv:
        # 处理单个CSV文件
        if not args.db:
            # 如果没有指定数据库路径，使用CSV文件同目录下的dataset.db
            csv_dir = os.path.dirname(args.csv)
            db_path = os.path.join(csv_dir, 'dataset.db')
        else:
            db_path = args.db
        process_single_csv(args.csv, db_path)
    elif args.all:
        # 处理所有CSV文件
        process_all_csv_files(args.data_dir)
    else:
        print("请指定 -csv 参数处理单个CSV文件，或使用 --all 处理所有CSV文件")
        print("使用 --help 查看详细帮助")

    args = parser.parse_args()
    
    if args.folder:
        # 处理包含gt.csv和dataset.db的文件夹
        folder_path = Path(args.folder)
        
        # 验证文件夹路径
        if not folder_path.exists():
            print(f"错误: 文件夹不存在: {folder_path}", file=sys.stderr)
            return 1
        
        # 查找gt.csv和dataset.db文件
        gt_csv_path = folder_path / "gt.csv"
        db_path = folder_path / "dataset.db"
        
        if not gt_csv_path.exists():
            print(f"错误: 在文件夹 {folder_path} 中未找到 gt.csv 文件", file=sys.stderr)
            return 1
        
        if not db_path.exists():
            print(f"错误: 在文件夹 {folder_path} 中未找到 dataset.db 文件", file=sys.stderr)
            return 1
        
        print(f"处理文件夹: {folder_path}")
        process_single_csv(str(gt_csv_path), str(db_path), args.verbose)
        
        # 如果需要详细信息，显示数据库表验证
        if args.verbose:
            verify_database_tables(str(db_path), args.verbose)
        
    elif args.csv:
        # 处理单个CSV文件
        if not args.db:
            # 如果没有指定数据库路径，使用CSV文件同目录下的dataset.db
            csv_dir = os.path.dirname(args.csv)
            db_path = os.path.join(csv_dir, 'dataset.db')
        else:
            db_path = args.db
        process_single_csv(args.csv, db_path, args.verbose)
        
        # 如果需要详细信息，显示数据库表验证
        if args.verbose:
            verify_database_tables(db_path, args.verbose)
            
    else:
        # 默认进行批处理
        process_batch_folders(args.data_dir, args.verbose)

if __name__ == "__main__":
    main()