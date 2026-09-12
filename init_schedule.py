import sqlite3
import os

DB_PATH = "schedule.db"

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE schedule (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day TEXT NOT NULL,
    time TEXT NOT NULL,
    course TEXT NOT NULL,
    location TEXT NOT NULL,
    teacher TEXT NOT NULL
)
""")

data = [
    ("周一", "08:00-09:40", "高等数学", "教学楼A101", "张老师"),
    ("周一", "10:00-11:40", "大学英语", "教学楼B203", "李老师"),
    ("周二", "14:00-15:40", "数据结构", "实验楼C305", "王老师"),
    ("周三", "08:00-09:40", "操作系统", "教学楼A202", "刘老师"),
    ("周三", "14:00-15:40", "计算机网络", "教学楼B105", "陈老师"),
    ("周四", "10:00-11:40", "线性代数", "教学楼A303", "赵老师"),
    ("周五", "08:00-09:40", "人工智能导论", "实验楼C401", "孙老师"),
]

cursor.executemany(
    "INSERT INTO schedule (day, time, course, location, teacher) VALUES (?, ?, ?, ?, ?)",
    data
)

conn.commit()
conn.close()
print(f"已创建 {DB_PATH}，插入 {len(data)} 条课程记录")