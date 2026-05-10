import shutil
from pathlib import Path

from base.types.navio_db import NaVIODB

target_path = Path(
    # "/Users/qi/Downloads/锐捷测试数据-2026-02-25/三星临时/dataset0226大厅21楼/dataset0226大厅21楼"
    # "/Users/qi/Downloads/锐捷测试数据-2026-02-25/三星临时/dataset0226图书馆21楼/dataset0226图书馆21楼"
    # "/Users/qi/Downloads/锐捷测试数据-2026-02-25/三星临时/dataset0226一楼门口/dataset0226一楼门口"
    "/Users/qi/Downloads/锐捷测试数据-2026-02-25/三星临时/dataset三星0225一楼大厅/dataset三星0225一楼大厅"
)

if __name__ == "__main__":
    dir_list = [dir for dir in target_path.iterdir() if dir.is_dir()]
    dir_list.sort()

    for dir in dir_list:
        # 将 db 复制一份，名称为 old，保留原文件
        db_path = dir / f"{dir.stem}.db"
        old_db_path = dir / f"{dir.stem}_old.db"
        if not db_path.exists():
            print(f"{dir.stem} is not exist.")
            continue

        if old_db_path.exists():
            print(f"{dir.stem} 已经处理过了")
            continue

        print(f"Process: {db_path}")
        # 从db中获取每一个pose，经过 convert 函数处理后再重新写入数据库
        db = NaVIODB(db_path=db_path)
        db.pose_fix_convert()

        shutil.copy(db_path, old_db_path)
        # 重新保存 db文件
        db.close()

    pass
